"""
Extract tables from PDFs stored in S3 buckets

SAMPLE USAGE:
cls & extract_tables.py --directory="D:\Dropbox\Projects\InflationData\macro_data_weimar\sources\Statistisches Jahrbuch\wages" --extension=pdf --bucket=quipucamayoc-tom

A FEW GOTCHAS:
- ClientRequestToken is based on filename but has undocumented requirement:
https://stackoverflow.com/a/64455772

```
"ClientRequestToken": {
  "type": "string",
  "max": 64,
  "min": 1,
  "pattern": "^[a-zA-Z0-9-_]+$"
},
```

"""

# ---------------------------
# Imports
# ---------------------------

import csv
import time
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import boto3
import click


# ---------------------------
# Functions
# ---------------------------

def get_client_request_token(document):
    text = Path(document).stem
    if text[0].isdigit():
        text = 'XX' + text
    text = re.sub('[^a-zA-Z0-9-_]', '_', text)
    text = 'PDF_' + text
    assert len(text) <= 64
    return text


def process_text_analysis(bucket, document, sns_topic_arn, role_arn):
    print('Starting job')
    textract_client = boto3.client('textract')
    # PDFs are located in the PDF folder
    # Also, I think the ClientRequestToken can't start with a number??
    print(f'{document=}')

    request_token = get_client_request_token(document)

    print(f'ClientRequestToken="{request_token}"')
    print(f'{bucket=}')
    print({'S3Object': {'Bucket': bucket, 'Name': document}})

    response = textract_client.start_document_analysis(
                    ClientRequestToken=request_token,
                    DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': document}},
                    FeatureTypes=['TABLES'],
                    JobTag='table',
                    NotificationChannel={'SNSTopicArn': sns_topic_arn,'RoleArn': role_arn})

    job_id = response['JobId']
    print(f'  {"JobId":<20}: "{job_id}"')
    for k, v in response['ResponseMetadata'].items():
        print(f'  {k:<20}: "{v}"')
    return job_id


def wait_for_completion(queue_url, job_id, output_path):
    print('Was the document already processed?')
    try:
        succeeded = download_and_save_tables(job_id, output_path)
        if succeeded:
            print(' - Yes!')
            return
        else:
            print(' - No; continuing')
    except boto3.client('textract').exceptions.InvalidJobIdException:
        print(' - No; continuing')  # pass

    print('Waiting for job completion')
    tic = time.process_time()
    job_found = False
    sqs_client = boto3.client('sqs')
    i = 0

    while not job_found:
        sqs_response = sqs_client.receive_message(QueueUrl=queue_url, MessageAttributeNames=['ALL'], MaxNumberOfMessages=10)
        if sqs_response:
            if 'Messages' not in sqs_response:
                if i < 40:
                    print('.', end='')
                    i += 1
                else:
                    print()
                    i = 0
                sys.stdout.flush()
                time.sleep(5)
                continue
            else:
                print()

            for message in sqs_response['Messages']:
                print(' - Message found')
                notification = json.loads(message['Body'])
                text_message = json.loads(notification['Message'])
                received_job_id = text_message['JobId']
                print(' -  Job ID from message:', received_job_id)
                print(' - Message status:', text_message['Status'])

                if job_id == received_job_id:
                    print('Matching Job Found:' + job_id)
                    job_found = True
                    toc = time.process_time()
                    print(f'Elapsed time: {toc - tic}')
                    download_and_save_tables(job_id, output_path)
                    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
                else:
                    print(f'Job didn\'t match: "{job_id}" vs "{received_job_id}"')
                    #sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
                


def download_and_save_tables(job_id, path):
    '''
    Download all the tables returned by Textract, and save them in the -path- folder with format:
    0123-09.tsv (table 9 of page 123)
    '''

    textract_client = boto3.client('textract')
    max_results = 1000  # Maximum number of blocks returned per request (1-1000)
    pagination_token = None  # Allows us to retrieve the next set of blocks (1-255)

    children = {}
    blocks_map = {}
    table_ids = []
    i = 0
    counter = Counter()

    while True:

        if pagination_token is None:
            response = textract_client.get_document_analysis(JobId=job_id, MaxResults=max_results)
            
            # Debugging?
            job_status = response['JobStatus']
            assert job_status in ('IN_PROGRESS', 'SUCCEEDED', 'FAILED'), job_status
            if job_status != 'SUCCEEDED':
                return False # Need to wait for the notification
            
            num_pages = response['DocumentMetadata']['Pages']
            print(f'Document text detected')
            print(f' - Pages: {num_pages}')
        else: 
            response = textract_client.get_document_analysis(JobId=job_id, MaxResults=max_results, NextToken=pagination_token)                           

        i += 1
        blocks = response['Blocks'] 
        print(f' - Part {i} has {len(blocks)} blocks')
        
        for block in blocks:

            block_id = block['Id']
            block_type = block['BlockType']
            blocks_map[block_id] = block

            counter[block_type] += 1

            # Debug info
            #if block_type == 'PAGE':
            #    print(block['Geometry'])

            if block_type == "TABLE":
                table_ids.append(block_id)

        if 'NextToken' in response:
            pagination_token = response['NextToken']
        else:
            print(' - All blocks downloaded')
            break

    print('Block types:')
    for block_type, n in counter.items():
        print(f' - Type "{block_type}": {n}')
    print()
    
    table_counter = Counter()
    for table_id in table_ids:
        # TODO: if results are unsorted, use Block['Geometry']['BoundingBox']['Top'] to sort and then save at the end
        
        ans = generate_table(table_id, blocks_map)
        page = blocks_map[table_id]['Page']
        table_counter[page] += 1
        table_num = table_counter[page]
        fn = path / f'{page:04}-{table_num:02}.tsv'

        with fn.open(mode='w', newline='', encoding='utf-8') as f:
            print(' - Saving file', fn.name)
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(ans)

    return True  # Success


def generate_table(table_id, blocks_map):
    rows = defaultdict(dict)
    table_result = blocks_map[table_id]
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    confidence = cell['Confidence']
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    text, warning = get_text(cell, blocks_map)
                    rows[row_index][col_index] = text
                    
                    if confidence < 30:
                        print(f'[Not sure about cell] text="{text}" conf={confidence}')
                    #elif warning:
                    #    print(f'[Unsure about some elements of cell] text="{text}" conf={confidence}')
    
    ans = [[text for j, text in sorted(cols.items())] for i, cols in sorted(rows.items())]
    return ans


def get_text(result, blocks_map):
    text = ''
    warning = False
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        confidence = word['Confidence']
                        word_text = word['Text']
                        if confidence < 30:
                            warning = True
                            print(f'[Not sure about word] text="{word_text}" conf={confidence}')
                            word_text = word_text + '???'
                        text += word_text + ' '
    return text, warning


def upload_pdf_if_needed(pdf_fn, bucket):
    s3_client = boto3.client('s3')
    document = f'pdf/{pdf_fn.name}' # PDFs are stored in the pdf folder within the S3 bucket
    print(f'Detecting if the bucket {bucket} already contains the key {document}')

    # Exists?
    results = s3_client.list_objects_v2(Bucket=bucket, Prefix=document, MaxKeys=1)
    key_exists = ('Contents' in results) and (document == results['Contents'][0]['Key'])

    if key_exists:
        print(' - Key already exists; continuing')
    else:
        print(' - Key does not exist; uploading')
        s3_client.upload_file(Filename=str(pdf_fn),
                              Bucket=bucket,
                              Key=document,
                              ExtraArgs={'StorageClass':'ONEZONE_IA'})
        print(' - PDF uploaded')

    return document


def delete_pdf(aws_fn, bucket):
    s3_client = boto3.client('s3')
    #document = f'pdf/{pdf_fn.name}' # PDFs are stored in the pdf folder within the S3 bucket
    document = aws_fn

    # Exists?
    results = s3_client.list_objects_v2(Bucket=bucket, Prefix=document, MaxKeys=1)
    key_exists = ('Contents' in results) and (document == results['Contents'][0]['Key'])

    if key_exists:
        s3_client.delete_object(Key=aws_fn, Bucket=bucket)
        print(f' - PDF deleted: {aws_fn}')
    else:
        print(f' - File "{aws_fn}" did not exist! Cannot delete!')


# ---------------------------
# Main
# ---------------------------

@click.command()
@click.option('-f', '--filename', '--file', type=str)
@click.option('-d', '--directory', '--dir', type=str)
@click.option('-e', '--extension', '--ext', type=str)
#@click.option('--bucket', default='quipucamayoc', type=str, help='AWS S3 bucket used (quipucamayoc, quipucamayoc-tom)')
@click.option('--keep/--no-keep', default=False, help='Keep object (PDF, PNG, or JPG) in AWS S3 bucket')

def main(filename, directory, extension, keep):

    print(f'{filename=}')
    print(f'{directory=}')
    bucket = 'quipu-bucket'

    num_options = (filename is not None) + (directory is not None)
    if num_options != 1:
        raise SystemExit("Error: must specify one and only one of --filename and --directory")
    if (directory is not None) and (extension is None):
        raise SystemExit("Error: --directory option requires --extension")
    if (extension is not None):
        assert extension in ('pdf', 'png', 'jpg')

    # ARN: Amazon resource name
    # SQS: Simple queue system
    # SNS: Simple notification system

    # View bucket contents here:
    # https://s3.console.aws.amazon.com/s3/buckets/quipucamayoc/pdf/?region=us-east-1
    
    #bucket = 'quipucamayoc'

    # Get list of files
    if directory is not None:
        directory = Path(directory)
        pdf_fns = directory.glob(f'*.{extension}')
    else:
        pdf_fn = Path(filename)
        if not pdf_fn.is_file():
            raise FileNotFoundError(str(pdf_fn))
        pdf_fns = [pdf_fn]
    
    for pdf_fn in pdf_fns:
        print(f'Processing file "{pdf_fn}"')

        #base_path = Path('E:/WH/textract')
        base_path = pdf_fn.parent
        output_path = base_path / pdf_fn.stem
        done_path = base_path / (pdf_fn.stem + '.done')

        if done_path.exists():
            print(f'File "{pdf_fn}" already processed; skipping')
            continue

        document = upload_pdf_if_needed(pdf_fn, bucket) # Output of document is "pdf/mypdf.pdf"
        output_path.mkdir(exist_ok=True)
        print(f'Document: {document}')

        if bucket == 'quipucamayoc':
            sns_topic_arn = 'arn:aws:sns:us-east-1:413472359132:AmazonTextractQuipucamayoc'
            role_arn = 'arn:aws:iam::413472359132:role/quipucamayoc-textract'
            queue_url = 'https://queue.amazonaws.com/413472359132/quipucamayoc_queue'
        else:
            sns_topic_arn = 'arn:aws:sns:us-east-1:835107504307:AmazonTextractQuipucamayoc'
            role_arn = 'arn:aws:iam::835107504307:role/quipucamayoc-textract'
            queue_url = 'https://sqs.us-east-1.amazonaws.com/835107504307/quipucamayoc_queue'


        job_id = process_text_analysis(bucket=bucket, document=document, sns_topic_arn=sns_topic_arn, role_arn=role_arn)
        wait_for_completion(queue_url, job_id, output_path)
        if not keep:
            delete_pdf(aws_fn=document, bucket=bucket)
        done_path.touch()

        print()
        print(f'File {pdf_fn} processed!')


if __name__ == "__main__":
    main()
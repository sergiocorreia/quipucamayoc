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

import re
import csv
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import boto3
from loguru import logger

from .aws_setup import QUIPU


# ---------------------------
# Functions
# ---------------------------

def filename2document(filename):
    assert isinstance(filename, Path)
    extension = filename.suffix[1:]
    assert extension in ('pdf', 'png', 'jpg', 'tiff'), extension  # TODO: handle unusual cases like .jpeg and .PDF (upper case)
    document = f'{extension}/{filename.name}'
    return document


def document_exists_in_bucket(document, bucket, s3_client, logger):
    logger.info(f' - Detecting if the bucket already contains key')
    results = s3_client.list_objects_v2(Bucket=bucket, Prefix=document, MaxKeys=1)
    key_exists = ('Contents' in results) and (document == results['Contents'][0]['Key'])
    if key_exists:
        logger.info(f' - Key exists')
    else:
        logger.info(f' - Key does not exist')
    return key_exists


def upload_file(filename, config, logger):
    '''
    Upload a filename to a bucket; don't do anything if file already in bucket
    '''
    
    document = filename2document(filename)
    bucket = config.bucket
    s3_client = config.session.client('s3')
    viewable_url = f'https://s3.console.aws.amazon.com/s3/buckets/{config.bucket}/{filename.suffix[1:]}/?region={config.region}'
    logger.info(f'Uploading file "{filename.name}" to bucket "{bucket}"')
    key_exists = document_exists_in_bucket(document, bucket, s3_client, logger)

    if not key_exists:
        logger.info(f' - Uploading...')
        s3_client.upload_file(Filename=str(filename), Bucket=bucket, Key=document, ExtraArgs={'StorageClass':'ONEZONE_IA'})  # ONEZONE_IA is cheaper
        logger.info(f' - File uploaded to {viewable_url}')

    return document


def delete_file(filename, config, logger):
    document = filename2document(filename)
    bucket = config.bucket
    s3_client = config.session.client('s3')
    logger.info(f'Deleting file "{filename.name}" from bucket "{bucket}"')
    key_exists = document_exists_in_bucket(document, bucket, s3_client, logger)

    if key_exists:
        s3_client.delete_object(Key=document, Bucket=bucket)
        logger.info(f' - File deleted')


def hash_file(filename):
    '''
    Hash used as ClientRequestToken for Textract
    '''

    # 1) Get salt from filename
    salt = filename.name
    salt = salt.replace('.', '_')
    salt = re.sub('[^a-zA-Z0-9-_]', '', salt)  # Required by ClientRequestToken
    salt = salt[:32]

    # 2) Get hash
    # See: https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    h = hashlib.sha256()
    bufsize = 128 * 1024 # Maybe make it bigger
    buffer = bytearray(bufsize)
    # using a memoryview so that we can slice the buffer without copying it
    buffer_view = memoryview(buffer)
    with filename.open('rb', buffering=0) as f:
        while n := f.readinto(buffer_view):
            h.update(buffer_view[:n])
    h.hexdigest()
    readable_hash = h.hexdigest()[:16] # Pick 8, 16, or maybe 32 (overkill)
    
    full_hash = f'{salt}-{readable_hash}'
    assert len(full_hash) <= 64
    return full_hash


def run_textract_async(filename, request_token, config, logger):
    # ClientRequestToken:
    #   - the idempotent token that you use to identify the start request.
    #   - Use ClientRequestToken to prevent the same job from being accidentally started more than once.
    #   - Warning: it seems (???) that the ClientRequestToken can't start with a number.
    # Also:
    #   - To prevent accidental duplication of analysis jobs, you can optionally provide an idempotent token, ClientRequestToken.
    #   - If you supply a value for ClientRequestToken, the Start operation returns the same JobId for multiple identical calls
    #     to the Start operation, such as StartDocumentTextDetection.
    #   - A ClientRequestToken token has a lifetime of 7 days. After 7 days, you can reuse it...
    # See also: https://docs.aws.amazon.com/textract/latest/dg/API_StartDocumentTextDetection.html

    # BUGBUG: what if a file is called "1.pdf" or "1.jpg" and we run this more than once in seven days? Maybe hash it?
    #Type: String
    #Length Constraints: Minimum length of 1. Maximum length of 64.
    #Pattern: ^[a-zA-Z0-9-_]+$

    
    logger.info(f'Starting async textract job for "{filename}"')
    document = filename2document(filename)
    textract_client = config.session.client('textract')
    logger.info(f' - Document = {document}')
    logger.info(f' - ClientRequestToken = {request_token}')

    response = textract_client.start_document_analysis(
                    ClientRequestToken=request_token,
                    DocumentLocation={'S3Object': {'Bucket': config.bucket, 'Name': document}},
                    FeatureTypes=['TABLES'],
                    JobTag='table',
                    NotificationChannel={'SNSTopicArn': config.topic_arn,'RoleArn': config.role_arn})

    #for k, v in response['ResponseMetadata'].items():
    #    print(f'  {k:<20}: "{v}"')
    job_id = response['JobId']
    logger.info(f' - JobID = {job_id}')
    return job_id


def wait(attempt, s):
    n=30
    if attempt % n == 0:
        if attempt:
            #newline every n dots
            print()
            return False
    else:
        print('.', end='')
        sys.stdout.flush()
        time.sleep(s) # TODO: add some sort of back-off to avoid polling so often (or use 5secs)
        return True


def wait_textract_async(filename, job_id, output_path, config, logger, output, page_append):

    logger.info('Waiting for job completion:')
    tic = time.perf_counter()
    sqs_client = config.session.client('sqs')
    max_seconds_waiting = 7200
    time_per_attempt = 3
    max_attempts = int(max_seconds_waiting / time_per_attempt) # 3600 * 2sec wait = 7200 sec

    try:
        succeeded, job_status = download_and_save_tables(job_id, output_path, config, logger, output, page_append)
        if succeeded:
            logger.warning(' - Job ID already existed on Textract; perhaps due to an early aborted run')
            return
        else:
            pass # logger.info(' - Job ID not yet on Textract')
    except config.session.client('textract').exceptions.InvalidJobIdException:
        logger.info(' - Job ID not yet on Textract (InvalidJobIdException)')

    logger.info(' - Waiting in queue for messages...')
    has_dots = False

    for attempt in range(max_attempts):   
        sqs_response = sqs_client.receive_message(QueueUrl=config.queue_url, MessageAttributeNames=['ALL'], MaxNumberOfMessages=10)
        if sqs_response:
            if 'Messages' not in sqs_response:
                has_dots = wait(attempt, time_per_attempt)
                continue
            elif has_dots:
                print()

            for message in sqs_response['Messages']:
                logger.info(' - Message found')
                notification = json.loads(message['Body'])
                text_message = json.loads(notification['Message'])
                received_job_id = text_message['JobId']
                logger.info(f'   Job ID from message: {received_job_id}')
                logger.info(f"   Message status: {text_message['Status']}")

                if job_id == received_job_id:
                    logger.success(' - Matching Job Found:' + job_id)
                    job_found = True
                    toc = time.perf_counter()
                    logger.info(f' - Elapsed time: {toc - tic:6.2f}s')
                    succeeded, job_status = download_and_save_tables(job_id, output_path, config, logger, output, page_append)
                    if not succeeded:
                        raise Exception(f"AWS Textract job status did not succeed (job-status='job_status')")
                    sqs_client.delete_message(QueueUrl=config.queue_url, ReceiptHandle=message['ReceiptHandle'])
                    return
                else:
                    logger.info(f'   Job didn\'t match: "{job_id}" vs "{received_job_id}"')
    #all attempts failed to return from function
    raise SystemExit("No response within response interval")


def save_to_file(object, path, logger, output):
    with path.open(mode='w', newline='', encoding='utf-8') as f:
        logger.info(f'   Saving file "{path.name}"')
        if output == "tsv":
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        elif output=="csv":
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        writer.writerows(object)


def save_tables_to_outputs(path, logger, output, table_ids, blocks_map):
    table_counter = Counter()

    for table_id in table_ids:
        logger.info(f' - Parsing table')
        # TODO: if results are unsorted, use Block['Geometry']['BoundingBox']['Top'] to sort and then save at the end
        
        ans = generate_table(table_id, blocks_map, logger)
        page = blocks_map[table_id]['Page']
        table_counter[page] += 1
        table_num = table_counter[page]
        fn = path / (f'{page:04}-{table_num:02}.' + output)
        save_to_file(ans, fn, logger, output)

        

def save_file_to_output(path, logger, output, table_ids, blocks_map):
    first_table = True
    full_ans = None
    for table_id in table_ids:
        logger.info(f' - Parsing table')
        
        if first_table:
            full_ans = generate_table(table_id, blocks_map, logger)
            first_table = False
        else: 
            #Cut header row
            ans = generate_table(table_id, blocks_map, logger)
            #Will need to be fine-tuned...
            # maybe flag first row that is not mostly full and mostly text
            ans = ans[2::]
            full_ans = full_ans + ans
    #TODO: when save_all_to_file enabled, set pathname earlier in pipeline
    fn = Path(str(path) + (f'.'+output))
    save_to_file(full_ans, fn, logger, output)


def download_and_save_tables(job_id, path, config, logger, output, page_append):
    '''
    Download all the tables returned by Textract, and save them in the -path- folder with format:
    0123-09.tsv (table 9 of page 123)
    TODO: for documents in multiple images, allow to save as {file}-{table}
    '''

    textract_client = config.session.client('textract')
    max_results = 1000  # Maximum number of blocks returned per request (1-1000)
    pagination_token = None  # Allows us to retrieve the next set of blocks (1-255)
    
    if output is None:
        output="tsv"

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
                return False, job_status # Need to wait for the notification
            
            num_pages = response['DocumentMetadata']['Pages']
            logger.info(f'Document text detected:')
            logger.info(f' - Pages: {num_pages}')
        else: 
            response = textract_client.get_document_analysis(JobId=job_id, MaxResults=max_results, NextToken=pagination_token)                           

        if not i:
            logger.info(f' - Downloading blocks...')
        i += 1
        blocks = response['Blocks'] 
        #logger.info(f' - Part {i} has {len(blocks)} blocks')
        
        for block in blocks:

            block_id = block['Id']
            block_type = block['BlockType']
            blocks_map[block_id] = block

            counter[block_type] += 1

            # Debug info
            #if block_type == 'PAGE':
            #    logger.info(block['Geometry'])

            if block_type == "TABLE":
                table_ids.append(block_id)

        if 'NextToken' in response:
            pagination_token = response['NextToken']
        else:
            logger.info(' - All blocks downloaded')
            break

    logger.info(' - Counts by block type:')
    for block_type, n in counter.items():
        logger.info(f'   - "{block_type}": {n}')

    if(page_append):
        save_file_to_output(path, logger, output, table_ids, blocks_map)
    else:
        save_tables_to_outputs(path, logger, output, table_ids, blocks_map)

    

    return True, 'SUCCEEDED'  # Success



def generate_table(table_id, blocks_map, logger):
    '''Generates as a [[]] list the table for the input'''
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
                        logger.debug(f'   @ Low confidence on cell="{text}" conf={confidence}')
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
                            logger.debug(f'   @ Low confidence on word="{word_text}" conf={confidence}')
                            word_text = word_text + '???'
                        text += word_text + ' '
    return text, warning


def aws_extract_from_file_helper(config, filename, keep_in_s3, ignore_cache, output, page_append):
    print(f'Extracting tables from {filename.name} with AWS Textract')
    output_path = filename.parent / ('textract-' + filename.stem)
    # won't overwrite existing directory
    output_path.mkdir(exist_ok=True)
    done_path = output_path / (filename.stem + '.done')
    is_done = done_path.is_file() and not ignore_cache

    if(page_append):
        output_path = output_path / filename.stem

    if is_done:
        logger.success(f'File "{filename}" already processed; skipping')
        exit()

    keep_in_s3 = True
        
    logger.info(f'Hashing file...')
    request_token = hash_file(filename)
    upload_file(filename, config, logger)
    job_id = run_textract_async(filename, request_token, config, logger)
    if not keep_in_s3:
        delete_file(filename, config, logger)
    wait_textract_async(filename, job_id, output_path, config, logger, output, page_append)

    done_path.touch()
    print()
    print(f'File "{filename}" processed!')


# ---------------------------
# Main function
# ---------------------------

def aws_extract_tables(filename=None, directory=None, extension=None, keep_in_s3=False, ignore_cache=False, output=None, page_append=False):

    # Logging details
    log_format = '<green>{time:HH:mm:ss.S}</green> | <level>{level: <8}</level> | <blue><level>{message}</level></blue>'
    logger.remove()
    logger.add(sys.stderr, format=log_format, colorize=True, level="INFO") # TRACE DEBUG INFO SUCCESS WARNING ERROR CRITICAL

    # Sanity checks
    num_options = (filename is not None) + (directory is not None)
    if num_options != 1:
        raise SystemExit("Error: must specify one and only one of --filename and --directory")
    if (directory is not None) and (extension is None):
        raise SystemExit("Error: --directory option requires --extension")
    if (extension is not None):
        assert extension in ('pdf', 'png', 'jpg', 'tiff')
    #Ideally more output formats to come!
    if (output is not None):
        assert output in ('csv','tsv')

    # Configuration details
    config = QUIPU()
    config.session = boto3.Session(profile_name='quipucamayoc', region_name=config.region) # Avoid specifying region_name for every client

    # [1] Single files (PDFs or images)
    if filename is not None:
        aws_extract_from_file(config, filename, keep_in_s3, ignore_cache, output, page_append)
        exit()


    # [2] and [3] Folders
    directory = Path(directory)
    filenames = directory.glob(f'*.{extension}')

    # [2] Folders with multiple PDFs
    if extension=='pdf':
        for filename in filenames:
            print(f'{filename=}')
            assert filename.is_file()
            aws_extract_from_file(config, filename, keep_in_s3, ignore_cache, output, page_append)
            raise SystemExit("QUIPUCAMAYOC INTERNAL ERROR: Incomplete function")
        exit()

    raise SystemExit("QUIPUCAMAYOC INTERNAL ERROR: Incomplete function")

    # [3] Folders with multiple images representing a single PDF
    for filename in filenames:
        print(f'{filename=}')
        assert filename.is_file()
        raise SystemExit("QUIPUCAMAYOC INTERNAL ERROR: Incomplete function")
"""
docstring

For boto3 documentation, see: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

Also see:
- https://hands-on.cloud/working-with-sns-in-python-using-boto3/

"""

# ---------------------------
# Imports
# ---------------------------

import sys
import time # not used
import json 
import logging # used just for -boto3- logging

import boto3
from loguru import logger
from pprint import pprint


# ---------------------------
# Constants
# ---------------------------


# ---------------------------
# Configuration Class
# ---------------------------

class QUIPU:
    region = 'us-east-1'
    user = 'Quipu'
    bucket = 'quipu-bucket'
    queue = 'quipu-queue'
    topic = 'AmazonTextractQuipu'
    role = 'quipu-textract'
    account_id = None
    topic_arn = None
    queue_arn = None
    queue_url = None
    subscription_arn = None
    role_arn = None

    # Sanity checks
    assert topic.startswith('AmazonTextract')


# ---------------------------
# Functions
# ---------------------------

def inspect(quipu, logger):
    # Configuration
    logger.info(f'Parameters:')
    parameters = {k:v for k,v in QUIPU.__dict__.items() if not k.startswith('__')}
    for k, v in parameters.items():
        logger.info(f'  - {k} = {v}')
    
    # Current user
    logger.info(f'Current AWS user: {QUIPU.account_id}')


def create_user(quipu, iam_client, logger):
    logger.info(f'Creating user "{quipu.user}"...')

    # Delete user if it already exists
    logger.info('  - Retrieving list of existing users')
    r = iam_client.list_users()
    assert not r['IsTruncated'] # Why would anyone have more than 100 users?
    users = [user['Arn'].split('/')[1] for user in r['Users']]
    logger.info(f'  - Found {len(users)} users')
    if quipu.user in users:
        logger.info(f'  - User "{quipu.user}" exists; deleting it')
        delete_user(quipu, iam_client, logger)
    else:
        logger.info(f'  - User "{quipu.user}" does not exist')

    # Set Permission boundary
    permissions_boundary = 'arn:aws:iam::aws:policy/PowerUserAccess'
    # Explanation:
    #   - This permission prevents the user to access IAM to limit risks
    #   - See: https://aws.amazon.com/premiumsupport/knowledge-center/iam-permission-boundaries/
    #   - Note: effective permissions are the _intersection_ of permission boundary and the defined identity-based policies
    #   - See slide 31 of: https://d1.awsstatic.com/events/reinvent/2019/REPEAT_1_AWS_identity_Permission_boundaries_&_delegation_SEC402-R1.pdf
    #   - List https://aws.amazon.com/blogs/security/how-to-assign-permissions-using-new-aws-managed-policies-for-job-functions/
    user = iam_client.create_user(UserName=quipu.user, PermissionsBoundary=permissions_boundary)


def delete_user(quipu, iam_client, logger):
    # We must detach existing INLINE policies before deleting user
    policies = iam_client.list_user_policies(UserName=quipu.user)['PolicyNames']
    for policy in policies:
        iam_client.delete_user_policy(UserName=quipu.user, PolicyName=policy)

    # We must detach existing MANAGED policies before deleting user
    policies = [p['PolicyArn'] for p in iam_client.list_attached_user_policies(UserName=quipu.user)['AttachedPolicies']]
    for policy in policies:
        iam_client.detach_user_policy(UserName=quipu.user, PolicyArn=policy)

    # We must delete access keys before deleting user
    keys = iam_client.list_access_keys(UserName=quipu.user)
    keys = [k['AccessKeyId'] for k in keys['AccessKeyMetadata']]
    for key in keys:
        iam_client.delete_access_key(UserName=quipu.user, AccessKeyId=key)

    # Now we can delete the user
    iam_client.delete_user(UserName=quipu.user)


def set_user_permissions(quipu, iam_client, logger):
    logger.info('Setting user permissions...')

    # Wait until change has propagated; unsure if really needed
    #waiter = iam_client.get_waiter('user_exists')
    #waiter.wait(UserName=quipu.user, WaiterConfig={'Delay': 1, 'MaxAttempts': 3})

    logger.info('  - Attaching managed policies (SQS, SNS, Textract)')
    iam_client.attach_user_policy(UserName=quipu.user, PolicyArn='arn:aws:iam::aws:policy/AmazonSQSFullAccess')
    iam_client.attach_user_policy(UserName=quipu.user, PolicyArn='arn:aws:iam::aws:policy/AmazonSNSFullAccess')
    iam_client.attach_user_policy(UserName=quipu.user, PolicyArn='arn:aws:iam::aws:policy/AmazonTextractFullAccess')

    logger.info('  - Creating inline policies (S3, Textract)')
    # BUGBUG MAKE THIS A DICT!!
    s3_policy = '''{
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "s3:*", "Resource": ["arn:aws:s3:::@", "arn:aws:s3:::@/*"]},
            {"Effect": "Deny", "NotAction": "s3:*", "NotResource": ["arn:aws:s3:::@", "arn:aws:s3:::@/*"]}
        ]}'''.replace('@', quipu.bucket)

    iam_client.put_user_policy(UserName=quipu.user, PolicyName='QuipuAllowS3', PolicyDocument=s3_policy)

    # Create custom inline policy
    statement = {'Sid': 'QuipuAssignRole',
                 'Action': 'iam:PassRole',
                 'Effect': 'Allow',
                 'Resource': quipu.role_arn}
    policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement]})
    iam_client.put_user_policy(UserName=quipu.user, PolicyName='QuipuAssignRole', PolicyDocument=policy_json)

    # Obtain and return credentials
    logger.info('Obtaining user credentials...')
    credentials = iam_client.create_access_key(UserName=quipu.user)['AccessKey']
    secret_access_key = credentials['SecretAccessKey']
    access_key_id = credentials['AccessKeyId']
    logger.info(f'  - Access Key ID = "{access_key_id}"')
    logger.info(f'  - Secret Access Key = "{secret_access_key}"')
    return secret_access_key, access_key_id


def create_bucket(quipu, s3_client, logger):
    logger.info(f'Creating bucket "{quipu.bucket}"')
    
    # Retrieve the list of existing buckets
    response = s3_client.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    
    # Delete bucket if needed
    if quipu.bucket in buckets:
        logger.info(f'  - Bucket "{quipu.bucket}" exists; deleting it')
        s3_client.delete_bucket(Bucket=quipu.bucket) # , ExpectedBucketOwner=quipu.user)
    
    # Workaround for AWS/BOTO3 bug:
    # If the location is us-east-1 (the default) then we CANNOT pass the location
    # See: https://github.com/boto/boto3/issues/125
    if quipu.region == 'us-east-1':
        s3_client.create_bucket(Bucket=quipu.bucket, ACL='private')
    else:
        s3_client.create_bucket(Bucket=quipu.bucket, ACL='private', CreateBucketConfiguration={'LocationConstraint': quipu.region})
    
    logger.info(f'  - Making bucket private')
    config = {'BlockPublicAcls': True, 'IgnorePublicAcls': True, 'BlockPublicPolicy': True, 'RestrictPublicBuckets': True}
    s3_client.put_public_access_block(Bucket=quipu.bucket, PublicAccessBlockConfiguration=config)


def create_sns_topic(quipu, sns_client, logger):
    logger.info(f'Creating SNS topic "{quipu.topic}"...')
    r = sns_client.create_topic(Name=quipu.topic)
    quipu.topic_arn = r['TopicArn']
    logger.info(f'  - SNS topic ARN = "{quipu.topic_arn}"')


def create_sqs_queue(quipu, sqs_client, logger):
    logger.info(f'Creating SQS standard queue "{quipu.queue}"...')

    logger.info('  - Retrieving list of existing queues')
    r = sqs_client.list_queues(QueueNamePrefix=quipu.queue)
    expected_queue_url = f'https://queue.amazonaws.com/{quipu.account_id}/{quipu.queue}'
    if False and 'QueueUrls' in r and expected_queue_url in r['QueueUrls']:
        logger.info(f'  - Queue "{quipu.queue}" exists; deleting it')
        r = sqs_client.delete_queue(QueueUrl=quipu.queue)  
        logger.info(f'  - Waiting 60 seconds before creating a new queue')
        # We have to wait 60s after deleting a queue to create a new one with the same name
        time.sleep(60)

    r = sqs_client.create_queue(QueueName=quipu.queue)

    # Fetch queue URL and ARN
    quipu.queue_url = r['QueueUrl']
    r = sqs_client.get_queue_attributes(QueueUrl=quipu.queue_url, AttributeNames=['QueueArn'])
    quipu.queue_arn = r['Attributes']['QueueArn']
    logger.info(f'  - SQS queue ARN = "{quipu.queue_arn}"')


def subscribe_queue_to_topic(quipu, sns_client, logger):
    logger.info(f'Subscribing SQS queue to SNS topic...')
    r = sns_client.subscribe(TopicArn=quipu.topic_arn, Protocol='sqs', Endpoint=quipu.queue_arn)
    quipu.subscription_arn = r['SubscriptionArn']
    logger.info(f'  - Subscription ARN = "{quipu.subscription_arn}"')


def allow_topic_in_queue(quipu, sqs_client, logger):
    logger.info(f'Giving SNS topic permission to message SQS queue...')
    statement = {'Sid': 'QuipuSendMessage',
                 'Action': 'SQS:SendMessage',
                 'Effect': 'Allow',
                 'Principal': {'AWS': '*'},  # TODO: Restrict to only QUIPU.account_id
                 'Resource': quipu.queue_arn,
                 'Condition': {'ArnEquals': {'aws:SourceArn': quipu.topic_arn}}}
    policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement]})
    sqs_client.set_queue_attributes(QueueUrl=quipu.queue_url, Attributes={'Policy': policy_json})


def create_iam_role(quipu, iam_client, logger):
    # See also: https://medium.com/geekculture/automating-aws-iam-using-lambda-and-boto3-part-3-3100088a4454
    logger.info(f'Creating IAM role so Textract can access SNS topics...')

    logger.info('  - Retrieving list of IAM roles')    
    r = iam_client.list_roles()
    roles = [role['RoleName'] for role in r['Roles']]
    if quipu.role in roles:
        logger.info(f'  - Role "{quipu.role}" exists; deleting it')
        role = iam_client.delete_role(RoleName=quipu.role)
    else:
        logger.info(f'  - Role "{quipu.role}" does not exist')

    statement = {'Effect': 'Allow',
                 'Principal': {'Service': 'textract.amazonaws.com'},
                 'Action': 'sts:AssumeRole'}
    policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement]})
    r = iam_client.create_role(RoleName=QUIPU.role,
                               AssumeRolePolicyDocument=policy_json,
                               Description='Allows AWS Textract to call other AWS services on your behalf')
    quipu.role_arn = r['Role']['Arn']



# ---------------------------
# Main
# ---------------------------

def setup_textract():
    print('Setting up AWS for quipucamayoc:')

    # Logging details
    #log_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    log_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <blue><level>{message}</level></blue>'
    #logger.remove()
    #logger.add(sys.stderr, format=log_format, colorize=True, level="DEBUG") # TRACE DEBUG INFO SUCCESS WARNING ERROR CRITICAL

    # Boto3 logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)8s | %(message)s') # DEBUG INFO

    # Show config details
    QUIPU.account_id = boto3.resource('iam').CurrentUser().arn.split(':')[4]
    inspect(QUIPU, logger)

    # Clients
    logger.info(f'Creating clients...')
    boto3.setup_default_session(region_name=QUIPU.region) # Avoid specifying region_name for every client
    iam_client = boto3.client('iam') #, region_name=QUIPU.region)
    s3_client = boto3.client('s3')
    sns_client = boto3.client('sns')
    sqs_client = boto3.client('sqs')

    # Create bucket
    create_bucket(QUIPU, s3_client, logger)

    # Create SNS topic
    create_sns_topic(QUIPU, sns_client, logger)

    # Create SQS standard queue
    create_sqs_queue(QUIPU, sqs_client, logger)

    # Subscribe SQS queue to SNS topic
    # >> In English: set up the notification service to message the queue once something happens
    subscribe_queue_to_topic(QUIPU, sns_client, logger)

    # Give permission to the Amazon SNS topic to send messages to the Amazon SQS queue.
    # >> In English: modify queue permissions so messages don't get rejected
    allow_topic_in_queue(QUIPU, sqs_client, logger)

    # Create an IAM service role to give Amazon Textract access to your Amazon SNS topics
    create_iam_role(QUIPU, iam_client, logger)

    # Create user, set its permissions, and obtain its credentials
    create_user(QUIPU, iam_client, logger)
    secret_access_key, access_key_id = set_user_permissions(QUIPU, iam_client, logger)


if __name__ == '__main__':
    setup_textract()

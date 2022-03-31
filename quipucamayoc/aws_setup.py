"""
docstring

For boto3 documentation, see: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

Also see:
- https://hands-on.cloud/working-with-sns-in-python-using-boto3/

AWS URLs:
- S3          https://s3.console.aws.amazon.com/s3/buckets/quipu-bucket
- SQS         https://console.aws.amazon.com/sqs/home?region=us-east-1#queue-browser:selected=https://sqs.us-east-1.amazonaws.com/413472359132/quipucamayoc_queue;prefix=
"""


# ---------------------------
# Imports
# ---------------------------

import os
import re
import sys
import errno
import time # not used
import json 
import configparser
import logging # used just for -boto3- logging
from pathlib import Path

import boto3, botocore
from botocore.exceptions import ClientError
from loguru import logger
#from pprint import pprint


# ---------------------------
# Constants
# ---------------------------


# ---------------------------
# Configuration Class
# ---------------------------

class QUIPU:

    user = 'Quipu'
    bucket_prefix = 'quipu-'
    queue = 'quipu-queue'
    topic = 'AmazonTextractQuipu'
    role = 'quipu-textract'
    credentials_section = 'quipucamayoc'


    def __init__(self, region=None):
        '''
        Fill in certain values (ARNs, URLs) used later by Textract
        
        Note:
         - ARN: Amazon resource name
         - SQS: Simple queue system
         - SNS: Simple notification system
        '''
        
        self.region = 'us-east-1' if region is None else region
        self.account_id = boto3.resource('iam').CurrentUser().arn.split(':')[4]
        self.topic_arn = f'arn:aws:sns:{self.region}:{self.account_id}:{self.topic}'
        self.role_arn = f'arn:aws:iam::{self.account_id}:role/{self.role}'
        self.queue_url = f'https://queue.amazonaws.com/{self.account_id}/{self.queue}'
        
        self.queue_arn = None
        self.subscription_arn = None

        # Problem: S3 buckets must be GLOBALLY UNIQUE
        # i.e., other users can't have use the same bucket name
        # To see if a bucket exists, go to https://<BUCKETNAME>.s3.amazonaws.com/
        # and see if the error message is AccessDenied or NoSuchBucket
        self.bucket = self.bucket_prefix + self.account_id

        # Sanity checks
        assert self.topic.startswith('AmazonTextract')
        self.validate_bucket()


    def validate_bucket(self):
        # https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
        # 1) Bucket names must be between 3 (min) and 63 (max) characters long.
        assert 3 <= len(self.bucket) <= 63
        # 2) Bucket names can consist only of lowercase letters, numbers, dots (.), and hyphens (-).
        invalid_characters = re.sub('[a-z0-9.-]', '', self.bucket)
        assert not invalid_characters
        # 3) Bucket names must begin and end with a letter or number.
        assert self.bucket[0] not in ('.', '-')
        assert self.bucket[-1] not in ('.', '-')





# ---------------------------
# Functions
# ---------------------------

def inspect(quipu, logger):
    # Configuration
    logger.info(f'Parameters:')
    parameters = {k:v for k,v in quipu.__dict__.items() if not k.startswith('__')}
    for k, v in parameters.items():
        logger.info(f'  - {k} = {v}')
    
    # Current user
    logger.info(f'Current AWS user: {quipu.account_id}')


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

    logger.info('  - Creating inline policies (S3, Textract role)')
    resources = [f'arn:aws:s3:::{quipu.bucket}', f'arn:aws:s3:::{quipu.bucket}/*']
    statement1 = {'Effect': 'Allow', 'Action': 's3:*', 'Resource': resources}
    statement2 = {'Effect': 'Deny', 'NotAction': 's3:*', 'NotResource': resources}
    policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement1]})
    #policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement1, statement2]}) # BUGBUG
    iam_client.put_user_policy(UserName=quipu.user, PolicyName='QuipuAllowS3', PolicyDocument=policy_json)

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
    access_key_id = credentials['AccessKeyId']
    secret_access_key = credentials['SecretAccessKey']
    logger.info(f'  - Access Key ID = "{access_key_id}"')
    logger.info(f'  - Secret Access Key = "{secret_access_key}"')
    return access_key_id, secret_access_key


def delete_bucket_and_contents(quipu, s3_client, logger, no_bucket_message=False):
    # Retrieve the list of existing buckets
    response = s3_client.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]

    # Delete bucket if needed
    if quipu.bucket in buckets:
        logger.info(f'  - Bucket "{quipu.bucket}" exists; deleting it')
        boto3.resource('s3').Bucket(quipu.bucket).objects.delete()
        s3_client.delete_bucket(Bucket=quipu.bucket) # , ExpectedBucketOwner=quipu.user)
    elif no_bucket_message:
        logger.info(f" - S3 bucket did not exist")


def create_bucket(quipu, s3_client, logger):
    logger.info(f'Creating bucket "{quipu.bucket}"')
    delete_bucket_and_contents(quipu, s3_client, logger)

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

def delete_sns_topic(quipu, sns_client, logger):
    topic_arn = f'arn:aws:sns:{quipu.region}:{quipu.account_id}:{quipu.topic}'
    r = sns_client.list_topics()
    topics = [topic['TopicArn'] for topic in r['Topics']]
    if topic_arn in topics:
        logger.info('  - Deleted SNS topic')
        sns_client.delete_topic(TopicArn=topic_arn)

    # Can't use list_subscriptions_by_topic() b/c topic might not exist anymore
    r = sns_client.list_subscriptions()
    subscriptions = [s['SubscriptionArn'] for s in r['Subscriptions'] if s['TopicArn']==topic_arn]
    if subscriptions:
        logger.info(f'  - Deleted {len(subscriptions)} SNS topic subscriptions')
        for subscription in subscriptions:
            boto3.resource('sns').Subscription(subscription).delete()


def create_sns_topic(quipu, sns_client, logger):
    logger.info(f'Creating SNS topic "{quipu.topic}"...')
    delete_sns_topic(quipu, sns_client, logger)
    r = sns_client.create_topic(Name=quipu.topic)
    assert quipu.topic_arn == r['TopicArn']
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
    assert quipu.queue_url == r['QueueUrl']
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
                 'Action': 'SQS:*', # 'Action': 'SQS:SendMessage',
                 'Effect': 'Allow',
                 'Principal': '*', #'Principal': {'AWS': '*'},  # TODO: Restrict to only quipu.account_id
                 'Resource': quipu.queue_arn,
                 'Condition': {'ArnEquals': {'aws:SourceArn': quipu.topic_arn}}}
    policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement]})
    sqs_client.set_queue_attributes(QueueUrl=quipu.queue_url, Attributes={'Policy': policy_json})


def delete_iam_role(quipu, iam_client, logger):
    logger.info('  - Retrieving list of IAM roles')    
    r = iam_client.list_roles()
    roles = [role['RoleName'] for role in r['Roles']]
    if quipu.role in roles:
        logger.info(f'  - Role "{quipu.role}" exists; deleting it')

        logger.info(f'    Detaching policies')
        policies = iam_client.list_attached_role_policies(RoleName=quipu.role)
        policies = [p['PolicyArn'] for p in policies['AttachedPolicies']]
        for policy in policies:
            iam_client.detach_role_policy(RoleName=quipu.role, PolicyArn=policy)
        
        logger.info(f'    Deleting role')
        iam_client.delete_role(RoleName=quipu.role)
    else:
        logger.info(f'  - Role "{quipu.role}" does not exist')


def create_iam_role(quipu, iam_client, logger):
    # See also: https://medium.com/geekculture/automating-aws-iam-using-lambda-and-boto3-part-3-3100088a4454
    logger.info(f'Creating IAM role so Textract can access SNS topics...')
    delete_iam_role(quipu, iam_client, logger)
    statement = {'Effect': 'Allow',
                 'Principal': {'Service': 'textract.amazonaws.com'},
                 'Action': 'sts:AssumeRole'}
    policy_json = json.dumps({'Version': '2012-10-17', 'Statement': [statement]})
    r = iam_client.create_role(RoleName=quipu.role,
                               AssumeRolePolicyDocument=policy_json,
                               Description='Allows AWS Textract to call other AWS services on your behalf')
    assert quipu.role_arn == r['Role']['Arn']

    # Attach managed textract policy to role
    iam_client.attach_role_policy(RoleName=quipu.role, PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonTextractServiceRole')


def add_credentials_to_file(quipu, access_key_id, secret_access_key, logger):
    config = configparser.ConfigParser()
    credentials_fn = Path(os.path.expanduser('~')) / '.aws' / 'credentials'
    logger.info('Adding new credentials to AWS credentials file')
    logger.info(f'  - File location: "{credentials_fn}"')

    if not credentials_fn.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(credentials_fn))

    logger.info('  - Reading current credentials')
    config.read(credentials_fn)

    if quipu.credentials_section not in config.sections():
        config.add_section(quipu.credentials_section)
    
    config['quipucamayoc']['aws_access_key_id'] = access_key_id
    config['quipucamayoc']['aws_secret_access_key'] = secret_access_key
    
    logger.info('  - Updating credentials')
    with open(credentials_fn, 'w') as f:
        config.write(f)


def delete_credentials(quipu):
    config = configparser.ConfigParser()
    credentials_fn = Path(os.path.expanduser('~')) / '.aws' / 'credentials'

    if not credentials_fn.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(credentials_fn))

    config.read(credentials_fn)

    if quipu.credentials_section in config.sections():
        config.remove_section(quipu.credentials_section)
        logger.info('  - Credentials removed from file')
        with open(credentials_fn, 'w') as f:
            config.write(f)
    else:
        logger.info('  - Credentials did not exist on file')


# ---------------------------
# Main functions
# ---------------------------

def install_aws():
    print('Setting up AWS for quipucamayoc:')

    # Logging details
    #log_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    #log_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <blue><level>{message}</level></blue>'
    log_format = '<green>{time:HH:mm:ss.S}</green> | <level>{level: <8}</level> | <blue><level>{message}</level></blue>'
    logger.remove()
    logger.add(sys.stderr, format=log_format, colorize=True, level="DEBUG") # TRACE DEBUG INFO SUCCESS WARNING ERROR CRITICAL

    # Boto3 logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)8s | %(message)s') # DEBUG INFO

    # Show config details
    quipu = QUIPU()  # No region passed so we'll use default
    inspect(quipu, logger)

    # Clients
    logger.info(f'Creating clients...')
    session = boto3.Session(profile_name='default', region_name=quipu.region) # Avoid specifying region_name for every client
    iam_client = session.client('iam')
    s3_client = session.client('s3')
    sns_client = session.client('sns')
    sqs_client = session.client('sqs')

    # Create bucket
    create_bucket(quipu, s3_client, logger)

    # Create SNS topic
    create_sns_topic(quipu, sns_client, logger)

    # Create SQS standard queue
    create_sqs_queue(quipu, sqs_client, logger)

    # Subscribe SQS queue to SNS topic
    # >> In English: set up the notification service to message the queue once something happens
    subscribe_queue_to_topic(quipu, sns_client, logger)

    # Give permission to the Amazon SNS topic to send messages to the Amazon SQS queue.
    # >> In English: modify queue permissions so messages don't get rejected
    allow_topic_in_queue(quipu, sqs_client, logger)

    # Create an IAM service role to give Amazon Textract access to your Amazon SNS topics
    create_iam_role(quipu, iam_client, logger)

    # Create user, set its permissions, and obtain its credentials
    create_user(quipu, iam_client, logger)
    access_key_id, secret_access_key = set_user_permissions(quipu, iam_client, logger)

    # Add credentials to file
    add_credentials_to_file(quipu, access_key_id, secret_access_key, logger)

    logger.success('AWS Textract pipeline successfully created!')


def uninstall_aws():
    '''Clear up all Quipucamayoc settings from AWS'''
    print('Cleaning up quipucamayoc from AWS:')

    log_format = '<green>{time:HH:mm:ss.S}</green> | <level>{level: <8}</level> | <blue><level>{message}</level></blue>'
    logger.remove()
    logger.add(sys.stderr, format=log_format, colorize=True, level="DEBUG") # TRACE DEBUG INFO SUCCESS WARNING ERROR CRITICAL

    quipu = QUIPU()

    # Clients
    session = boto3.Session(profile_name='default', region_name=quipu.region) # Avoid specifying region_name for every client
    iam_client = session.client('iam')
    s3_client = session.client('s3')
    sns_client = session.client('sns')
    sqs_client = session.client('sqs')

    # Delete SNS topic
    delete_sns_topic(quipu, sns_client, logger)

    # Delete SQS queue
    try:
        sqs_client.delete_queue(QueueUrl=quipu.queue)
        logger.info('  - Deleted SQS queue')
    except ClientError as e:
        logger.info(f"  - SQS queue did not exist ({e.response['Error']['Code']})")

    # Delete IAM role
    try:
        delete_iam_role(quipu, iam_client, logger)
        logger.info('  - Deleted IAM role')
    except ClientError as e:
        logger.info(f"  - IAM role did not exist ({e.response['Error']['Code']})")

    # Delete user
    try:
        delete_user(quipu, iam_client, logger)
        logger.info('  - Deleted user')
    except ClientError as e:
        logger.info(f"  - User did not exist ({e.response['Error']['Code']})")

    # Delete bucket
    delete_bucket_and_contents(quipu, s3_client, logger, no_bucket_message=True)
    #try:
    #    s3_client.delete_bucket(Bucket=quipu.bucket)
    #except ClientError as e:
    #    error_code = e.response['Error']['Code']
    #    if error_code == '???BucketNotEmpty???':
    #        print(f" - S3 bucket did not exist ({e.response['Error']['Code']})")
    #    else:
    #        raise e

    # Delete credentials
    delete_credentials(quipu)
    logger.success(' - All quipucamayoc-related settings removed from AWS account')


# ---------------------------
# Main section
# ---------------------------

if __name__ == '__main__':
    #setup_textract()
    #install_aws()
    pass

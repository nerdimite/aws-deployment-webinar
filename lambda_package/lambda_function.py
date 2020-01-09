import json
import re
import boto3

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

with open('vocab_dict.json', 'r') as fp:
    VOCAB_DICT = json.load(fp)


def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

def preprocess_input(text, vocab_dict, maxlen=100):
    review = review_to_words(text)
    tokens = review.split()
    int_tokens = [vocab_dict[token] for token in tokens]
    
    if len(int_tokens) >= maxlen:
        return int_tokens[:maxlen]
    else:
        diff = maxlen - len(int_tokens)
        zeros = [0 for i in range(diff)]
        return [zeros + int_tokens]

def lambda_handler(event, context):
    
    data = preprocess_input(event['body'], VOCAB_DICT)

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = 'sagemaker-tensorflow-2020-01-09-00-43-42-678',    # The name of the endpoint we created
                                       ContentType = 'text/plain',                 # The data format that is expected
                                       Body = json.dumps(data))                       # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = json.loads(response['Body'].read().decode("utf-8"))
    pred = result['outputs']['score']['floatVal'][0]

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : pred
    }
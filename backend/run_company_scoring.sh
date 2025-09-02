#!/bin/bash
set -e

# -----------------------------
# Variables - UPDATE THESE
# -----------------------------
AWS_REGION="us-east-2"
ACCOUNT_ID="149536460887"
ECR_REPO_NAME="company-scoring"
JOB_DEF_NAME="company-scoring"
JOB_QUEUE="company-scoring-queue"
JOB_NAME="company-scoring-latest"
DOCKERFILE_PATH="."   # directory containing Dockerfile
PYTHON_SCRIPT="/app/company_scoring.py"
VCPUS=4
MEMORY=10240
JOB_ROLE_ARN="arn:aws:iam::149536460887:role/AWSBatchExecutionRole"
EXEC_ROLE_ARN="arn:aws:iam::149536460887:role/AWSBatchExecutionRole"
# -----------------------------

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push Docker image for linux/amd64
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
echo "Building Docker image for linux/amd64..."
docker buildx build --platform linux/amd64 -t $IMAGE_URI $DOCKERFILE_PATH --push

# Create container-properties JSON
CONTAINER_JSON=$(mktemp)
cat > $CONTAINER_JSON <<EOF
{
  "image": "$IMAGE_URI",
  "vcpus": $VCPUS,
  "memory": $MEMORY,
  "command": ["python", "$PYTHON_SCRIPT"],
  "jobRoleArn": "$JOB_ROLE_ARN",
  "executionRoleArn": "$EXEC_ROLE_ARN",
  "networkConfiguration": {
    "assignPublicIp": "ENABLED"
  },
  "fargatePlatformConfiguration": {
    "platformVersion": "LATEST"
  },
  "runtimePlatform": {
    "cpuArchitecture": "X86_64",
    "operatingSystemFamily": "LINUX"
  }
}
EOF

# Register new job definition
echo "Registering AWS Batch job definition..."
aws batch register-job-definition \
    --job-definition-name $JOB_DEF_NAME \
    --type container \
    --platform-capabilities FARGATE
    --container-properties file://$CONTAINER_JSON

# Submit job
echo "Submitting job to queue..."
aws batch submit-job \
    --job-name $JOB_NAME \
    --job-queue $JOB_QUEUE \
    --job-definition $JOB_DEF_NAME

echo "Job submitted successfully!"

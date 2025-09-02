# Company Scoring Backend - AWS Batch Deployment Guide

Welcome to the Company Scoring project! This guide will walk you through the complete process of updating the Python script, packaging it as a Docker image, pushing it to AWS, and submitting it as a job to AWS Batch.

## 📋 Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Docker Desktop** - For building container images
- **AWS CLI** - For interacting with AWS services
- **Python 3.11+** - For local development and testing
- **AWS Account Access** - With permissions for ECR, Batch, and IAM

### AWS CLI Configuration
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and default region (us-east-2), ** check Notion for access keys **
```

## 🏗️ Project Structure

```
backend/
├── company_scoring.py          # Main Python script for company scoring
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── run_company_scoring.sh      # Deployment automation script
├── container-properties.json   # AWS Batch job configuration
├── req_script.py               # Utility to auto-generate requirements.txt (You wouldn't have to run it unless you import a new library to company_scoring.py)
└── venv/                       # Virtual environment (local development)
```

## 🚀 Step-by-Step Deployment Process

### Step 1: Update the Python Script

1. **Make your changes** to `company_scoring.py`
   - The script performs company scoring analysis using machine learning
   - It reads data from S3, processes it, and outputs results
   - Key functions include data preprocessing, PCA, clustering, and scoring

2. **Update dependencies** (if needed):
   ```bash
   # Option A: Auto-generate requirements from imports
   python req_script.py
   
   # Option B: Manually edit requirements.txt
   # Add any new packages you've imported
   ```

3. **Test locally** (recommended):
   ```bash
   # Activate virtual environment
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the script locally to test
   python company_scoring.py
   ```

### Step 2: Build and Push Docker Image

The `run_company_scoring.sh` script automates the entire deployment process. Here's what it does:

#### Configuration Variables (Update if needed)
Edit the variables at the top of `run_company_scoring.sh`:

```bash
AWS_REGION="us-east-2"                    # AWS region
ACCOUNT_ID="149536460887"                 # Your AWS account ID
ECR_REPO_NAME="company-scoring"           # ECR repository name
JOB_DEF_NAME="company-scoring"            # Batch job definition name
JOB_QUEUE="company-scoring-queue"         # Batch job queue name
JOB_NAME="company-scoring-latest"         # Job name (change for each run)
VCPUS=4                                   # CPU cores for the job
MEMORY=10240                              # Memory in MB (10GB)
```

#### Run the Deployment Script
```bash
# Make the script executable
chmod +x run_company_scoring.sh

# Run the deployment
./run_company_scoring.sh
```

### Step 3: What the Script Does

The deployment script performs these steps automatically:

1. **ECR Login**: Authenticates Docker with AWS ECR
2. **Build & Push**: Creates a Docker image for linux/amd64 platform and pushes to ECR
3. **Job Definition**: Registers a new AWS Batch job definition with container properties
4. **Job Submission**: Submits the job to the specified queue

#### Manual Steps (if you prefer to run individually):

```bash
# 1. Login to ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 149536460887.dkr.ecr.us-east-2.amazonaws.com

# 2. Build and push Docker image
docker buildx build --platform linux/amd64 -t 149536460887.dkr.ecr.us-east-2.amazonaws.com/company-scoring:latest . --push

# 3. Register job definition
aws batch register-job-definition \
    --job-definition-name company-scoring \
    --type container \
    --platform-capabilities FARGATE \
    --container-properties file://container-properties.json

# 4. Submit job
aws batch submit-job \
    --job-name company-scoring-latest \
    --job-queue company-scoring-queue \
    --job-definition company-scoring
```

## 🔧 Understanding the Components

### Dockerfile
- Uses Python 3.11 slim base image
- Copies the Python script and requirements
- Installs dependencies
- Sets up the working environment

### Container Properties
The `container-properties.json` defines:
- **Resources**: 4 vCPUs, 10GB memory
- **Platform**: AWS Fargate with X86_64 architecture
- **Networking**: Public IP enabled
- **IAM Roles**: Execution and job roles for AWS permissions

### AWS Batch Configuration
- **Job Queue**: `company-scoring-queue` (must exist in AWS Batch)
- **Job Definition**: Containerized job definition
- **Platform**: AWS Fargate (serverless compute)

## 📊 Monitoring Your Job

### Check Job Status
```bash
# List recent jobs
aws batch list-jobs --job-queue company-scoring-queue

# Get job details
aws batch describe-jobs --jobs <job-id>
```

### View Logs
1. Go to AWS CloudWatch Console
2. Navigate to Log Groups
3. Look for `/aws/batch/job` log group
4. Find your job's log stream

## 🛠️ Troubleshooting

### Common Issues

1. **ECR Login Failed**
   - Verify AWS credentials: `aws sts get-caller-identity`
   - Check region configuration

2. **Docker Build Failed**
   - Ensure Docker Desktop is running
   - Check Dockerfile syntax
   - Verify all files are present

3. **Job Submission Failed**
   - Verify job queue exists: `aws batch describe-job-queues`
   - Check IAM permissions for Batch service
   - Ensure ECR repository exists

4. **Job Execution Failed**
   - Check CloudWatch logs for error details
   - Verify S3 bucket permissions
   - Check if input data exists in S3

### Useful Commands

```bash
# Check AWS configuration
aws sts get-caller-identity

# List ECR repositories
aws ecr describe-repositories

# List Batch job queues
aws batch describe-job-queues

# List job definitions
aws batch describe-job-definitions --job-definition-name company-scoring
```

## 🔄 Development Workflow

1. **Make changes** to `company_scoring.py`
2. **Test locally** with sample data
3. **Update requirements.txt** if needed
4. **Run deployment script** to push to AWS
5. **Monitor job execution** in AWS Console
6. **Check results** in S3 or CloudWatch logs

## 📝 Notes for New Interns

- Always test your changes locally before deploying
- Use meaningful job names (e.g., include date/time)
- Monitor resource usage - adjust vCPUs/memory if needed
- Keep the ECR repository clean by removing old images periodically
- The script processes company data from S3 bucket `good-companies`
- Results are typically saved back to S3 or logged to CloudWatch

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review AWS CloudWatch logs for detailed error messages
3. Verify all AWS resources (ECR repo, Batch queue, IAM roles) exist
4. Ask your team lead for assistance with AWS permissions or configuration

---

**Happy coding! 🚀**

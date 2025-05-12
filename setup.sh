#!/bin/bash

# Setup script for ML API project
# This script prepares the project for deployment

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_header() { echo -e "${BLUE}[SETUP]${NC} $1"; }

# Check if we're in the right directory
if [ ! -d "terraform" ] || [ ! -d "ml-api" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

print_header "Setting up ML API project..."

# 1. Create missing directories
print_status "Creating missing directories..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p config/dev

# 2. Copy existing VPC module
print_header "Setting up networking module..."
if [ -d "Project1" ]; then
    print_status "Copying existing Terraform VPC code..."
    cp Project1/main.tf terraform/modules/networking/ 2>/dev/null || print_warning "main.tf already exists"
    cp Project1/variables.tf terraform/modules/networking/ 2>/dev/null || print_warning "variables.tf already exists"
else
    print_warning "Project1 directory not found. Please copy your VPC Terraform code manually."
fi

# 3. Create dummy models
print_header "Creating dummy models for testing..."
if command -v python3 &> /dev/null; then
    cd scripts && python3 create_dummy_models.py && cd ..
else
    print_warning "Python3 not found. Please run scripts/create_dummy_models.py manually."
fi

# 4. Create development tfvars file
print_header "Creating development configuration..."
cat > terraform/environments/dev/terraform.tfvars.example << EOF
# Development Environment Configuration
aws_region = "us-east-1"
project_name = "ml-api-dev"
vpc_name = "ml-api-dev-vpc"
vpc_cidr = "10.0.0.0/16"

# Subnets configuration
private_subnets = {
  "private_subnet_1" = 1
  "private_subnet_2" = 2
}

public_subnets = {
  "public_subnet_1" = 1
  "public_subnet_2" = 2
}

# Container configuration
container_port = 8000

# Optional: Domain name for custom domain
# domain_name = "api.yourdomain.com"
EOF

# Copy to actual tfvars if it doesn't exist
if [ ! -f "terraform/environments/dev/terraform.tfvars" ]; then
    cp terraform/environments/dev/terraform.tfvars.example terraform/environments/dev/terraform.tfvars
    print_status "Created terraform.tfvars file"
fi

# 5. Create environment file for local development
print_header "Creating environment configuration..."
cat > ml-api/.env.example << EOF
# Development Environment Variables
ENVIRONMENT=development
DEBUG=true
MODELS_PATH=/app/data/models
LOG_LEVEL=INFO

# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET_NAME=

# API Configuration
API_V1_PREFIX=/v1
CORS_ORIGINS=*

# Optional: External services
REDIS_URL=redis://localhost:6379
EOF

# 6. Make scripts executable
print_header "Setting up scripts..."
chmod +x scripts/build/build_docker.sh
chmod +x scripts/deploy/deploy_dev.sh
chmod +x scripts/create_dummy_models.py

# 7. Create gitignore additions if needed
print_status "Checking .gitignore..."
if ! grep -q "terraform.tfvars" .gitignore; then
    echo -e "\n# Terraform\nterraform.tfvars" >> .gitignore
fi

if ! grep -q ".env" .gitignore; then
    echo -e "\n# Environment files\n.env" >> .gitignore
fi

# 8. Create quick start guide
print_header "Creating quick start guide..."
cat > QUICKSTART.md << EOF
# ML API Quick Start Guide

## Prerequisites
- Docker installed
- AWS CLI configured
- Terraform installed
- Python 3.9+ (for creating dummy models)

## Quick Setup

### 1. Project Setup
\`\`\`bash
# The setup script has already run!
# If you need to run it again:
./setup.sh
\`\`\`

### 2. Local Development
\`\`\`bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r ml-api/requirements.txt

# Run locally
cd ml-api
uvicorn app.main:app --reload
\`\`\`

### 3. Docker Development
\`\`\`bash
# Build and run with Docker Compose
docker-compose up --build
\`\`\`

### 4. Deploy to AWS
\`\`\`bash
# Deploy everything
./scripts/deploy/deploy_dev.sh

# Or deploy in stages
./scripts/deploy/deploy_dev.sh --only-infra  # Infrastructure only
./scripts/deploy/deploy_dev.sh --only-app    # Application only
\`\`\`

## API Endpoints

Once deployed, your API will have these endpoints:
- \`GET /\` - Root endpoint
- \`GET /health\` - Health check
- \`GET /docs\` - Interactive API documentation
- \`POST /v1/predict/academic-performance\` - Predict academic impact
- \`POST /v1/predict/mental-health\` - Predict mental health score
- \`POST /v1/predict/sleep-pattern\` - Predict sleep hours
- \`POST /v1/predict/addiction-risk\` - Calculate addiction risk

## What's Next?

1. Replace dummy models with real trained models
2. Add your actual dataset to \`data/raw/\`
3. Run the EDA notebook: \`notebooks/01_data_exploration.ipynb\`
4. Train your models and save them to \`data/models/\`
5. Update the API endpoints as needed

## Monitoring

After deployment, check:
- CloudWatch Dashboard: Get URL from deployment script output
- ECS Service: AWS Console > ECS > Clusters
- API Health: \`curl <your-alb-url>/health\`
EOF

print_status "Created QUICKSTART.md"

# Summary
print_header "Setup Complete! 🎉"
echo
print_status "What was done:"
echo "✅ Created project directory structure"
echo "✅ Set up Terraform configurations"
echo "✅ Created dummy ML models for testing"
echo "✅ Made scripts executable"
echo "✅ Created configuration files"
echo "✅ Generated quick start guide"
echo
print_status "Next steps:"
echo "1. Review and edit terraform/environments/dev/terraform.tfvars"
echo "2. Run: ./scripts/deploy/deploy_dev.sh"
echo "3. Check QUICKSTART.md for detailed instructions"
echo
print_warning "Note: This setup uses dummy models. Replace them with your actual models when ready."
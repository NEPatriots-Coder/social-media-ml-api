#!/bin/bash

# Deploy to Development Environment
# Complete deployment script for ML API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script configuration
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME=${PROJECT_NAME:-ml-api-dev}

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_header() { echo -e "${BLUE}[DEPLOY]${NC} $1"; }

# Check requirements
check_requirements() {
    print_header "Checking requirements..."
    
    for cmd in terraform aws docker; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        exit 1
    fi
    
    print_status "All requirements satisfied"
}

# Deploy infrastructure
deploy_infrastructure() {
    print_header "Deploying infrastructure..."
    
    cd terraform/environments/$ENVIRONMENT
    
    # Initialize terraform
    print_status "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    print_status "Planning infrastructure changes..."
    terraform plan -out=tfplan
    
    # Apply changes
    print_status "Applying infrastructure changes..."
    terraform apply tfplan
    
    # Save outputs for later use
    terraform output -json > ../../../terraform-outputs.json
    
    cd ../../..
    print_status "Infrastructure deployment complete"
}

# Build and push Docker image
build_and_push_image() {
    print_header "Building and pushing Docker image..."
    
    # Get ECR repository URL from Terraform output
    ECR_URL=$(cat terraform-outputs.json | jq -r '.ecr_repository_url.value')
    
    if [ "$ECR_URL" == "null" ]; then
        print_error "Could not get ECR repository URL from Terraform output"
        exit 1
    fi
    
    # Login to ECR
    print_status "Logging into ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URL
    
    # Build image
    print_status "Building Docker image..."
    docker build -t $PROJECT_NAME:latest ./ml-api
    
    # Tag for ECR
    docker tag $PROJECT_NAME:latest $ECR_URL:latest
    
    # Push to ECR
    print_status "Pushing image to ECR..."
    docker push $ECR_URL:latest
    
    print_status "Docker image built and pushed successfully"
}

# Update ECS service
update_ecs_service() {
    print_header "Updating ECS service..."
    
    # Get service details from Terraform output
    CLUSTER_NAME=$(cat terraform-outputs.json | jq -r '.ecs_cluster_name.value')
    SERVICE_NAME=$(cat terraform-outputs.json | jq -r '.ecs_service_name.value')
    
    # Force new deployment
    print_status "Forcing new deployment..."
    aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --force-new-deployment --region $AWS_REGION
    
    # Wait for deployment to complete
    print_status "Waiting for service to become stable..."
    aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION
    
    print_status "ECS service updated successfully"
}

# Show deployment info
show_deployment_info() {
    print_header "Deployment Information"
    
    ALB_URL=$(cat terraform-outputs.json | jq -r '.application_url.value')
    API_DOCS_URL=$(cat terraform-outputs.json | jq -r '.api_documentation_url.value')
    DASHBOARD_URL=$(cat terraform-outputs.json | jq -r '.cloudwatch_dashboard_url.value')
    
    echo -e "\nDeployment Complete! 🚀"
    echo -e "\nApplication URL: ${GREEN}$ALB_URL${NC}"
    echo -e "API Documentation: ${GREEN}$API_DOCS_URL${NC}"
    echo -e "CloudWatch Dashboard: ${GREEN}$DASHBOARD_URL${NC}"
    echo -e "\nHealth Check: curl $ALB_URL/health"
    echo -e "API Test: curl $ALB_URL/v1/predict/academic-performance"
}

# Main execution
main() {
    print_header "=== ML API Deployment Script ==="
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --region)
                AWS_REGION="$2"
                shift 2
                ;;
            --project-name)
                PROJECT_NAME="$2"
                shift 2
                ;;
            --only-infra)
                ONLY_INFRA=true
                shift
                ;;
            --only-app)
                ONLY_APP=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --environment   Target environment (default: dev)"
                echo "  --region        AWS region (default: us-east-1)"
                echo "  --project-name  Project name (default: ml-api-dev)"
                echo "  --only-infra    Deploy only infrastructure"
                echo "  --only-app      Deploy only application (skip infrastructure)"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    check_requirements
    
    if [ "$ONLY_APP" != "true" ]; then
        deploy_infrastructure
    fi
    
    if [ "$ONLY_INFRA" != "true" ]; then
        build_and_push_image
        update_ecs_service
    fi
    
    show_deployment_info
    
    print_status "Deployment completed successfully! 🎉"
}

# Run main function
main "$@"
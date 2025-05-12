# Social Media Usage Impact Predictor API

A machine learning API that predicts various impacts of social media usage on students, including academic performance, mental health scores, and addiction risk levels.

## Architecture

- **Infrastructure**: AWS ECS/Fargate with Application Load Balancer
- **API Framework**: FastAPI with Pydantic models
- **ML Stack**: scikit-learn, pandas, statsmodels
- **Infrastructure as Code**: Terraform
- **Container Technology**: Docker

## Project Structure

```
social-media-ml-api/
├── terraform/              # Infrastructure as Code
│   ├── modules/            # Reusable Terraform modules
│   └── environments/       # Environment-specific configurations
├── ml-api/                 # Python API application
│   ├── app/               # FastAPI application code
│   └── scripts/           # Utility scripts
├── notebooks/             # Jupyter notebooks for data science
├── data/                  # Data storage
└── docs/                  # Documentation
```

## Models Deployed

1. **Academic Performance Classifier**: Predicts if social media affects academic performance
2. **Mental Health Score Predictor**: Estimates mental health score based on usage patterns
3. **Sleep Pattern Analyzer**: Predicts sleep hours based on social media habits
4. **Addiction Risk Calculator**: Multi-class classification for addiction levels

## Quick Start

### Prerequisites

- AWS CLI configured
- Terraform >= 1.0
- Docker
- Python 3.9+

### Local Development

1. Clone the repository
2. Set up the environment:
   ```bash
   cd ml-api
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   uvicorn app.main:app --reload
   ```

### Deployment

1. Deploy infrastructure:
   ```bash
   cd terraform/environments/dev
   terraform init
   terraform plan
   terraform apply
   ```

2. Build and deploy containers:
   ```bash
   ./scripts/deploy/deploy_dev.sh
   ```

## API Endpoints

- `POST /v1/predict/academic-performance`: Predict academic impact
- `POST /v1/predict/mental-health`: Predict mental health score
- `POST /v1/predict/sleep-pattern`: Predict sleep hours
- `POST /v1/predict/addiction-risk`: Calculate addiction risk level
- `GET /docs`: Interactive API documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
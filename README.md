# Intelligent Data Quality & Drift Monitoring System

**Production-Grade MLOps Pipeline with Automated Drift Detection and Model Registry**

## Executive Summary

This is a containerized MLOps system designed for production deployment of machine learning models with continuous monitoring capabilities. It implements the complete ML lifecycle: training, versioning, deployment, and drift detection with automated alerting.

**Core Value Proposition:**
- Detects data distribution shifts before they degrade model performance
- Maintains reproducible training environments and versioned artifacts
- Provides centralized experiment tracking and model governance
- Enables safe model rollbacks through versioned registry
- Reduces model degradation incidents through proactive monitoring

**Use Case:** Any production ML system requiring data quality assurance and performance stability over time (fraud detection, recommendation systems, predictive maintenance, etc.)

---

## System Architecture

### High-Level Design

```
┌────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                  │
│                                                                │
│  ┌──────────────────┐         ┌─────────────────────────────┐  │
│  │  MLflow Server   │         │   Training Pipeline         │  │
│  │  (Port 5001)     │◄────────│   - Data processing         │  │
│  │                  │  HTTP   │   - CatBoost training       │  │
│  │  - Tracking API  │         │   - Hyperparameter tuning   │  │
│  │  - Model Registry│         │   - Metric logging          │  │
│  │  - SQLite Backend│         │   - Model registration      │  │
│  └────────┬─────────┘         └─────────────────────────────┘  │
│           │                                                    │
│           │ Shared Volume                                      │
│           │ (mlflow_data)                                      │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Artifact Storage                           │   │
│  │  - Trained models (pickle/CatBoost native)              │   │
│  │  - Reference datasets (production baseline)             │   │
│  │  - Preprocessor objects                                 │   │
│  │  - Experiment metadata                                  │   │
│  └────────┬────────────────────────────────────────────────┘   │
│           │                                                    │
│           │ HTTP API                                           │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          Monitoring Pipeline                            │   │
│  │  - Load registered model from registry                  │   │
│  │  - Fetch reference dataset from artifacts               │   │
│  │  - Compare production data vs reference                 │   │
│  │  - Generate Evidently drift reports                     │   │
│  │  - Trigger alerts on threshold violations               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘

External Interfaces:
- MLflow UI: http://localhost:5001
- Training: docker-compose run training
- Monitoring: docker-compose run monitoring
```

### Data Flow Architecture

```
┌─────────────┐
│  Raw Data   │
│  (CSV/DB)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Training Pipeline         │
│                             │
│  1. Data Validation         │
│  2. Feature Engineering     │
│  3. Train/Test Split        │
│  4. Model Training          │
│  5. Evaluation              │
└──────┬──────────────────────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌──────────────┐         ┌─────────────────┐
│ MLflow       │         │ Reference       │
│ Artifacts    │         │ Dataset         │
│              │         │ (Baseline)      │
│ - Model      │         └────────┬────────┘
│ - Metrics    │                  │
│ - Params     │                  │
└──────┬───────┘                  │
       │                          │
       ▼                          │
┌──────────────┐                  │
│ Model        │                  │
│ Registry     │                  │
│              │                  │
│ Production   │                  │
│ Staging      │                  │
│ Archived     │                  │
└──────┬───────┘                  │
       │                          │
       │                          │
       ▼                          ▼
┌─────────────────────────────────────────-┐
│      Monitoring Pipeline                 │
│                                          │
│  1. Load production model                │
│  2. Receive new production data          │
│  3. Compare vs reference dataset         │
│  4. Calculate drift metrics              │
│  5. Generate alerts if drift > threshold │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ Alert System │
│ - Logs       │
│ - Reports    │
│ - Metrics    │
└──────────────┘
```

---

## Component Breakdown

### 1. MLflow Tracking Server

**Purpose:** Centralized experiment tracking and model registry with persistent storage

**Implementation:**
- Runs as standalone service in Docker
- SQLite backend for metadata (runs, experiments, parameters, metrics)
- Local filesystem for artifact storage (models, datasets, preprocessing objects)
- HTTP API exposed on port 5001
- Single-writer architecture prevents SQLite locking issues

**Key Operations:**
```python
# Training pipeline interaction
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.log_params(hyperparameters)
mlflow.log_metrics(evaluation_results)
mlflow.log_artifact(model_path)

# Monitoring pipeline interaction
client = MlflowClient("http://mlflow:5001")
model = mlflow.pyfunc.load_model(model_uri)
```

**Production Considerations:**
- SQLite is sufficient for single-team usage; migrate to PostgreSQL for multi-team/high-concurrency
- Artifact storage should be migrated to S3/GCS/Azure Blob for production scale
- Current setup handles ~100 experiments/day comfortably

### 2. Training Pipeline

**Purpose:** Reproducible model training with experiment tracking and versioning

**Workflow:**
1. **Data Ingestion:** Load raw data from configured source
2. **Validation:** Schema validation, null checks, type verification
3. **Feature Engineering:** Automated feature creation, encoding, scaling
4. **Splitting:** Stratified train/test split with configurable ratio
5. **Training:** CatBoost with hyperparameter optimization
6. **Evaluation:** Calculate metrics (accuracy, precision, recall, F1, AUC-ROC)
7. **Logging:** Push all artifacts and metrics to MLflow
8. **Registration:** Promote model to registry if performance criteria met

**Key Design Decisions:**

**CatBoost Selection:**
- Handles categorical features natively (no manual encoding)
- Built-in handling of missing values
- Fast inference (critical for production)
- Excellent default parameters (reduces tuning time)

**Reference Dataset Storage:**
- Training data statistics saved as MLflow artifact
- Used as baseline for drift detection
- Enables reproducible monitoring across model versions

**Model Versioning Strategy:**
```
Run ID → Experiment → Model Version → Registry Stage
(unique) (grouping)  (sequential)    (Production/Staging/Archived)
```

### 3. Model Registry

**Purpose:** Model lifecycle management and version control

**Registry Stages:**
- **None:** Newly logged models (pending review)
- **Staging:** Models undergoing validation/testing
- **Production:** Currently deployed models
- **Archived:** Deprecated models (retained for rollback)

**Version Management:**
```python
# Automatic versioning on registration
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="credit_risk_classifier"
)

# Stage transitions
client.transition_model_version_stage(
    name="credit_risk_classifier",
    version=3,
    stage="Production"
)
```

**Rollback Capability:**
- All historical versions retained
- Can revert to any previous production model
- Artifact storage preserves complete model state

### 4. Monitoring Pipeline

**Purpose:** Continuous data quality and drift detection with automated alerting

**Monitoring Capabilities:**

**Data Drift Detection (Evidently 0.6):**
- Statistical tests per feature (Kolmogorov-Smirnov, chi-squared, etc.)
- Distribution comparison (reference vs current)
- Drift score calculation (0-1 scale)
- Column-level drift identification

**Model Performance Monitoring:**
- Prediction distribution shifts
- Class imbalance changes
- Feature importance drift

**Alert Conditions:**
```python
ALERT_THRESHOLDS = {
    'dataset_drift': 0.3,      # >30% features drifted
    'feature_drift': 0.5,      # Individual feature drift score
    'prediction_drift': 0.4    # Output distribution shift
}
```

**Workflow:**
1. Load latest production model from registry
2. Retrieve reference dataset from artifacts
3. Accept new production data (batch or streaming)
4. Generate Evidently drift report
5. Parse drift metrics
6. Compare against thresholds
7. Log alerts and generate reports

**Production Integration Points:**
- Can be scheduled via cron/Airflow/Kubernetes CronJob
- Accepts data from REST API, message queue, or data lake
- Outputs to logging system, Slack, PagerDuty, etc.

---

## Folder Structure

```
mlops-project/
│
├── docker-compose.yml              # Orchestration configuration
├── Dockerfile.training             # Training pipeline container
├── Dockerfile.monitoring           # Monitoring pipeline container
│
├── mlflow/
│   └── Dockerfile                  # MLflow server container
│
├── training/
│   ├── train.py                    # Main training orchestrator
│   ├── data_processing.py          # Feature engineering, validation
│   ├── model.py                    # CatBoost wrapper and utilities
│   ├── config.py                   # Hyperparameters, paths, constants
│   └── requirements.txt            # Training dependencies
│
├── monitoring/
│   ├── monitor.py                  # Drift detection orchestrator
│   ├── drift_detection.py          # Evidently integration
│   ├── alerting.py                 # Alert logic and thresholds
│   ├── config.py                   # Monitoring configuration
│   └── requirements.txt            # Monitoring dependencies
│
├── data/
│   ├── raw/                        # Source data (not in git)
│   ├── processed/                  # Transformed data
│   └── reference/                  # Baseline for drift detection
│
├── mlflow_data/                    # Persistent volume for MLflow
│   ├── mlflow.db                   # SQLite tracking database
│   └── artifacts/                  # Model binaries, datasets
│
└── reports/                        # Generated drift reports
    └── drift_report_YYYYMMDD.html
```

**Organizational Rationale:**

**Separation of Concerns:**
- Training and monitoring are independent services
- Each has isolated dependencies (separate requirements.txt)
- Enables independent deployment and scaling

**Configuration Management:**
- `config.py` in each service for environment-specific settings
- Avoids hardcoded values
- Supports dev/staging/prod environments

**Data Isolation:**
- Raw data never modified in place
- Processed data versioned separately
- Reference data persisted for reproducibility

---

## Production-Grade Characteristics

### 1. Reproducibility
- Dockerized environment ensures consistent runtime
- Pinned dependencies (Evidently==0.6)
- MLflow tracks all experiments, parameters, and artifacts
- Reference datasets versioned alongside models

### 2. Observability
- All training runs logged with metrics and parameters
- Drift monitoring provides continuous visibility
- MLflow UI for experiment comparison
- Alert system for proactive issue detection

### 3. Version Control
- Model registry tracks all versions
- Stage-based promotion workflow
- Rollback capability via registry
- Git-based code versioning

### 4. Fault Tolerance
- SQLite single-writer prevents database corruption
- Container restart policies in docker-compose
- Graceful error handling in pipelines
- Artifact persistence via volumes

### 5. Scalability Considerations
- Training pipeline can be scheduled via orchestrator
- Monitoring pipeline supports batch or streaming data
- MLflow backend can be upgraded to PostgreSQL
- Artifact storage can be moved to object storage

### 6. Security & Governance
- Model registry provides audit trail
- Access control can be added via MLflow authentication
- Data validation prevents malformed inputs
- Alert system ensures compliance monitoring

---

## Key Production Decisions & Trade-offs

### Technology Choices

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **CatBoost** | Native categorical handling, fast inference, good defaults | Less interpretable than linear models |
| **MLflow** | Industry standard, mature, strong community | Not as feature-rich as commercial tools (SageMaker, Vertex AI) |
| **Evidently 0.6** | Open-source, statistical rigor, HTML reports | Version 0.6 pinned due to API stability (newer versions differ) |
| **SQLite** | Zero-config, sufficient for single writer | Not suitable for high-concurrency (use Postgres at scale) |
| **Docker Compose** | Simple orchestration, local development | Not for production Kubernetes (use Helm charts for K8s) |

### Design Trade-offs

**Batch vs Streaming Monitoring:**
- **Current:** Batch processing (scheduled runs)
- **Alternative:** Real-time streaming (Kafka + Flink)
- **Reasoning:** Batch is simpler, sufficient for most use cases (daily/hourly monitoring)

**Model Serving:**
- **Current:** Model stored in registry, loaded by monitoring pipeline
- **Alternative:** Separate serving layer (TensorFlow Serving, Seldon, BentoML)
- **Reasoning:** Monitoring focuses on drift detection, not high-throughput inference

**Alert System:**
- **Current:** Log-based alerts
- **Alternative:** Integration with PagerDuty, Slack, email
- **Reasoning:** Extensible foundation; production deployment adds integrations

---

## How to Run

### Prerequisites
```bash
# Required
docker >= 20.10
docker-compose >= 1.29
```

### Initial Setup
```bash
# Clone repository
git clone <repo-url>
cd mlops-project

# Create data directories
mkdir -p data/raw data/processed data/reference mlflow_data reports

# Place training data in data/raw/
# Expected: CSV file with features and target column
```

### Start MLflow Server
```bash
docker-compose up -d mlflow

# Verify server is running
curl http://localhost:5001/api/2.0/mlflow/experiments/list

# Access UI
open http://localhost:5001
```

### Train Model
```bash
# Run training pipeline
docker-compose run --rm training

# Expected output:
# - Model logged to MLflow
# - Metrics displayed in console
# - Model registered in registry
# - Reference dataset saved to artifacts
```

### Monitor for Drift
```bash
# Run monitoring pipeline
docker-compose run --rm monitoring

# Expected output:
# - Drift report generated in reports/
# - Alerts logged if drift detected
# - Metrics pushed to MLflow
```

### View Results
```bash
# MLflow UI: Compare experiments, view metrics
http://localhost:5001

# Drift Reports: Open HTML files
open reports/drift_report_YYYYMMDD.html
```

### Cleanup
```bash
# Stop all services
docker-compose down

# Remove volumes (deletes MLflow data)
docker-compose down -v
```

---

## Mapping to Industry Roles

### ML Engineer
- Model training pipeline design
- Hyperparameter tuning and experimentation
- Feature engineering automation
- Model evaluation and selection

### MLOps Engineer
- Dockerized deployment architecture
- CI/CD integration (extendable with GitHub Actions/Jenkins)
- MLflow server configuration and management
- Monitoring pipeline implementation

### Data Engineer
- Data validation and quality checks
- Reference dataset management
- Pipeline orchestration (Airflow integration ready)
- Data versioning strategy

### ML Platform Engineer
- Model registry design and governance
- Artifact storage management
- Scalability considerations (Kubernetes migration path)
- Infrastructure as code (docker-compose → Terraform)

---

## Extension Points

### For Production Deployment

**Infrastructure:**
- [ ] Migrate docker-compose to Kubernetes (Helm charts)
- [ ] Replace SQLite with PostgreSQL for MLflow backend
- [ ] Use S3/GCS for artifact storage
- [ ] Add Redis for caching and job queuing

**Monitoring:**
- [ ] Integrate Prometheus for metrics collection
- [ ] Add Grafana dashboards for visualization
- [ ] Implement real-time alerting (PagerDuty, Slack)
- [ ] Model performance monitoring (actual vs predicted)

**Automation:**
- [ ] Add CI/CD pipeline (GitHub Actions, GitLab CI)
- [ ] Automated testing (unit, integration, model tests)
- [ ] Scheduled training and monitoring (Airflow DAGs)
- [ ] Automated model promotion based on metrics

**Security:**
- [ ] MLflow authentication and authorization
- [ ] Secrets management (Vault, AWS Secrets Manager)
- [ ] API authentication for monitoring endpoints
- [ ] Data encryption at rest and in transit

**Advanced Features:**
- [ ] A/B testing framework
- [ ] Multi-model comparison
- [ ] Automated retraining triggers
- [ ] Model explainability integration (SHAP, LIME)

---

## Conclusion

This system demonstrates production-ready MLOps practices:
- **Reproducible training** through containerization and experiment tracking
- **Model governance** via versioned registry with stage-based promotion
- **Proactive monitoring** with automated drift detection and alerting
- **Operational simplicity** while maintaining extensibility for enterprise scale

The architecture is designed for real-world deployment, not academic demonstration. All components are industry-standard tools used in production ML systems at scale.

---

## License

MIT

## Contact

For questions or contributions, please open an issue or submit a pull request.
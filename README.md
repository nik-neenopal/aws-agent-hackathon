# AWS Agent Hackathon: Metadata Enrichment Pipeline

## Overview

An intelligent, serverless metadata enrichment pipeline that leverages AWS services and AI agents to process, analyze, and enrich file metadata. This project demonstrates the power of combining event-driven architecture with AI agents powered by Amazon Bedrock and Claude Haiku to automate data processing workflows.

##  Architecture

The system is built on AWS Cloud infrastructure utilizing a modern serverless architecture that automatically processes uploaded files, enriches their metadata using AI agents, and stores structured data for downstream consumption.

### Key Components

**Frontend Layer**
- **CloudFront + S3**: Static website hosting for user interface
- **API Gateway**: RESTful API endpoints for file upload and data retrieval

**Processing Pipeline**
- **Lambda Functions**: Serverless compute for event handling and orchestration
- **Amazon S3**: Object storage with event notifications for triggering workflows
- **Amazon RDS**: Dual-database architecture with original and staging tables for data validation

**AI Agent Core**
- **Bedrock Claude Haiku**: AI model for intelligent metadata extraction and enrichment
- **Agent Core Runtime**: Orchestration engine for AI agent workflows

**Data Flow**
- **System Input S3**: Repository for processed metadata and system artifacts

**Supported Services**
- **IAM**: Fine-grained access control and security
- **CloudWatch Logs**: Centralized logging and monitoring

##  Data Flow

1. **File Upload**: Users upload files through the CloudFront-hosted frontend
2. **API Processing**: API Gateway receives the upload and triggers the first Lambda function
3. **S3 Event**: File stored in S3 triggers the metadata enrichment pipeline
4. **AI Processing**: Agent Core Runtime with Bedrock Claude Haiku analyzes and enriches file metadata
5. **Data Storage**: Processed outputs stored in both RDS (original and staging tables) and S3
6. **Validation & Finalization**: Staging data validated and promoted to the fixed table
7. **Frontend Delivery**: API Gateway Lambda sends final enriched data back to the frontend

##  Features

- **Serverless Architecture**: Fully managed, auto-scaling infrastructure
- **AI-Powered Enrichment**: Intelligent metadata extraction using Claude Haiku
- **Event-Driven Processing**: Automatic pipeline triggering based on S3 events
- **Data Validation**: Staging environment for quality control before production
- **Real-Time Monitoring**: CloudWatch integration for observability
- **Secure Access**: IAM-based security model

##  Technology Stack

- **Compute**: AWS Lambda
- **Storage**: Amazon S3, Amazon RDS
- **AI/ML**: Amazon Bedrock (anthropic.claude-3-haiku-20240307-v1:0)
- **API**: Amazon API Gateway
- **Frontend**: Amazon CloudFront, S3 (Static Website)
- **Security**: AWS IAM
- **Monitoring**: Amazon CloudWatch

## Project Structure

```
aws-agent-hackathon/
├── services/
│   ├── lambdas/          # Lambda function implementations
│   │   └── readme.md     # Lambda service documentation
│   └── agent-core/       # AI agent runtime and orchestration
│       └── readme.md     # Agent core documentation
├── frontend/             # CloudFront static website
```


## Documentation

- [Lambda Functions Documentation](./services/lambdas/readme.md)
- [Agent Core Documentation](./services/agent-core/readme.md)
- [Frontend Documentation](././frontend/csv-agent-frontend/README.md)
- [Confidence Score Documentation](./services/agent-core/ConfidenceScore.md)

## Security

- All AWS resources follow least-privilege IAM policies
- Data encrypted at rest (S3, RDS) and in transit (TLS/HTTPS)
- VPC isolation for RDS databases
- CloudWatch logging enabled for audit trails

---

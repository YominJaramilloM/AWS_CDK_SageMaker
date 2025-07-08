from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker_cfn,
    RemovalPolicy
)
from aws_cdk import aws_sagemaker_alpha as sagemaker_alpha
from constructs import Construct
import json

class MpsGroupStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Buckets S3
        self.raw_data_bucket = s3.Bucket(
            self, "RawDataBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )
        
        self.processed_data_bucket = s3.Bucket(
            self, "ProcessedDataBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Rol de SageMaker
        self.sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ]
        )

        # Grupo de paquetes de modelo
        self.model_package_group = sagemaker_cfn.CfnModelPackageGroup(
            self, "ModelPackageGroup",
            model_package_group_name="mlops-mps-model-group",
            model_package_group_description="Group for registering trained models"
        )

        # Definición del pipeline
        pipeline_definition = {
            "Version": "2020-12-01",
            "Steps": [
                {
                    "Name": "PreprocessingStep",
                    "Type": "Processing",
                    "Arguments": {
                        "ProcessingJobName": "mlops-mps-processing-job",
                        "AppSpecification": {
                            "ImageUri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1",
                            "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/processing.py"]
                        },
                        "RoleArn": self.sagemaker_role.role_arn,
                        "ProcessingInputs": [
                            {
                                "InputName": "input-1",
                                "S3Input": {
                                    "S3Uri": f"s3://{self.raw_data_bucket.bucket_name}/data.csv",
                                    "LocalPath": "/opt/ml/processing/input",
                                    "S3DataType": "S3Prefix",
                                    "S3InputMode": "File",
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3CompressionType": "None"
                                }
                            },
                            {
                                "InputName": "code",
                                "S3Input": {
                                    "S3Uri": f"s3://{self.raw_data_bucket.bucket_name}/processing.py",
                                    "LocalPath": "/opt/ml/processing/code",
                                    "S3DataType": "S3Prefix",
                                    "S3InputMode": "File",
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3CompressionType": "None"
                                }
                            }
                        ],
                        "ProcessingOutputConfig": {
                            "Outputs": [
                                {
                                    "OutputName": "output-1",
                                    "S3Output": {
                                        "S3Uri": f"s3://{self.processed_data_bucket.bucket_name}/processed/",
                                        "LocalPath": "/opt/ml/processing/output",
                                        "S3UploadMode": "EndOfJob"
                                    }
                                }
                            ]
                        },
                        "ProcessingResources": {
                            "ClusterConfig": {
                                "InstanceType": "ml.m5.large",
                                "InstanceCount": 1,
                                "VolumeSizeInGB": 20
                            }
                        },
                        "StoppingCondition": {
                            "MaxRuntimeInSeconds": 600
                        }
                    }
                },
                {
                "Name": "TrainingStep",
                "Type": "Training",
                "Arguments": {
                    "TrainingJobName": "mlops-mps-training-job",
                    "AlgorithmSpecification": {
                        "TrainingImage": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1",
                        "TrainingInputMode": "File",
                    },
                    "RoleArn": self.sagemaker_role.role_arn,
                    "HyperParameters": {
                        "sagemaker_program": "train.py",
                        "sagemaker_submit_directory": f"s3://{self.processed_data_bucket.bucket_name}/code.tar.gz"
                    },
                    "InputDataConfig": [
                        {
                            "ChannelName": "train",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": f"s3://{self.processed_data_bucket.bucket_name}/processed/"
                                }
                            },
                            "ContentType": "text/csv"
                        },
                        {
                            "ChannelName": "code",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": f"s3://{self.processed_data_bucket.bucket_name}/code.tar.gz"
                                }
                            },
                            "ContentType": "application/x-python",
                            "InputMode": "File"
                        }
                    ],
                    "OutputDataConfig": {
                        "S3OutputPath": f"s3://{self.processed_data_bucket.bucket_name}/model/"
                    },
                    "ResourceConfig": {
                        "InstanceType": "ml.m5.large",
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 30
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": 3600
                    }
                }
                },
                {
                    "Name": "RegisterModel",
                    "Type": "RegisterModel",
                    "Arguments": {
                        "ModelPackageGroupName": self.model_package_group.model_package_group_name,
                        "InferenceSpecification": {
                            "Containers": [
                                {
                                    "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1",
                                    "ModelDataUrl": {
                                        "Get": "Steps.TrainingStep.ModelArtifacts.S3ModelArtifacts"
                                    },
                                    "Environment": {
                                        "SAGEMAKER_PROGRAM": "inference.py",
                                        "SAGEMAKER_SUBMIT_DIRECTORY": {
                                            "Get": "Steps.TrainingStep.ModelArtifacts.S3ModelArtifacts"
                                        }
                                    }
                                }
                            ],
                            "SupportedContentTypes": ["text/csv"],
                            "SupportedResponseMIMETypes": ["text/csv"]
                        }
                    }
                }
            ]
        }
        print(json.dumps(pipeline_definition, indent=2))
        # Creación del pipeline
        self.pipeline = sagemaker_cfn.CfnPipeline(
            self, "SageMakerPipeline",
            pipeline_name="mlops-mps-pipeline-V3",
            role_arn=self.sagemaker_role.role_arn,
            pipeline_definition={
                "PipelineDefinitionBody": json.dumps(pipeline_definition, indent=2)
            }
        )
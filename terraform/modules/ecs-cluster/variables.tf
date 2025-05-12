# ECS Cluster Module Variables

variable "cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
  default     = "ml-api-cluster"
}

variable "service_name" {
  description = "Name of the ECS service"
  type        = string
  default     = "ml-api-service"
}

variable "task_family" {
  description = "Task definition family name"
  type        = string
  default     = "ml-api-task"
}

variable "container_name" {
  description = "Name of the container"
  type        = string
  default     = "ml-api"
}

variable "container_image" {
  description = "Container image URI"
  type        = string
}

variable "container_port" {
  description = "Port exposed by the container"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "CPU units for the task"
  type        = string
  default     = "512"
}

variable "memory" {
  description = "Memory for the task in MB"
  type        = string
  default     = "1024"
}

variable "desired_count" {
  description = "Desired count of tasks"
  type        = number
  default     = 2
}

variable "min_capacity" {
  description = "Minimum capacity for auto scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum capacity for auto scaling"
  type        = number
  default     = 10
}

variable "cpu_target_value" {
  description = "Target CPU utilization for auto scaling"
  type        = number
  default     = 70.0
}

variable "security_group_ids" {
  description = "Security group IDs for the service"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for the service"
  type        = list(string)
}

variable "target_group_arn" {
  description = "ALB target group ARN"
  type        = string
}

variable "alb_listener_arn" {
  description = "ALB listener ARN"
  type        = string
}

variable "enable_container_insights" {
  description = "Enable container insights for the cluster"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 14
}

variable "environment_variables" {
  description = "Environment variables for the container"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

variable "s3_bucket_arn" {
  description = "S3 bucket ARN for model storage"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
# ALB Module Variables

variable "alb_name" {
  description = "Name of the Application Load Balancer"
  type        = string
  default     = "ml-api-alb"
}

variable "internal" {
  description = "Whether the ALB is internal or internet-facing"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "VPC ID where the ALB will be created"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for the ALB"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for the ALB"
  type        = list(string)
}

variable "target_group_name" {
  description = "Name of the target group"
  type        = string
  default     = "ml-api-tg"
}

variable "target_port" {
  description = "Port for the target group"
  type        = number
  default     = 8000
}

variable "health_check_path" {
  description = "Health check path"
  type        = string
  default     = "/health"
}

variable "certificate_arn" {
  description = "SSL certificate ARN (optional)"
  type        = string
  default     = ""
}

variable "ssl_policy" {
  description = "SSL policy for HTTPS listener"
  type        = string
  default     = "ELBSecurityPolicy-TLS-1-2-2017-01"
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = false
}

variable "enable_access_logs" {
  description = "Enable access logs to S3"
  type        = bool
  default     = false
}

variable "access_logs_retention_days" {
  description = "Number of days to retain access logs"
  type        = number
  default     = 30
}

variable "response_time_threshold" {
  description = "Response time threshold for CloudWatch alarm (seconds)"
  type        = number
  default     = 1.0
}

variable "error_5xx_threshold" {
  description = "5xx error count threshold for CloudWatch alarm"
  type        = number
  default     = 10
}

variable "alarm_actions" {
  description = "Actions to take when CloudWatch alarms are triggered"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
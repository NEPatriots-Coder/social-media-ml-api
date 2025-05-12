# ALB Module Outputs

output "alb_id" {
  description = "ALB ID"
  value       = aws_lb.main.id
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.main.arn
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB zone ID"
  value       = aws_lb.main.zone_id
}

output "target_group_id" {
  description = "Target group ID"
  value       = aws_lb_target_group.app.id
}

output "target_group_arn" {
  description = "Target group ARN"
  value       = aws_lb_target_group.app.arn
}

output "target_group_name" {
  description = "Target group name"
  value       = aws_lb_target_group.app.name
}

output "http_listener_arn" {
  description = "HTTP listener ARN"
  value       = var.certificate_arn != "" ? aws_lb_listener.http.arn : aws_lb_listener.http_only[0].arn
}

output "https_listener_arn" {
  description = "HTTPS listener ARN (if SSL enabled)"
  value       = var.certificate_arn != "" ? aws_lb_listener.https[0].arn : null
}

output "alb_security_group_ids" {
  description = "Security group IDs attached to ALB"
  value       = var.security_group_ids
}

output "s3_bucket_name" {
  description = "S3 bucket name for ALB access logs (if enabled)"
  value       = var.enable_access_logs ? aws_s3_bucket.alb_logs[0].bucket : null
}
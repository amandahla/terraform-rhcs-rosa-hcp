module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.6.1"

  name = "production-vpc"
  cidr = "10.0.0.0/16"
}

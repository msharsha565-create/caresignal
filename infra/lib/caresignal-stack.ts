import * as path from "path";
import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

/**
 * CareSignal on AWS: one Fargate service (API + agent + MCP server + built UI)
 * behind an Application Load Balancer, calling Claude through Amazon Bedrock
 * with the task's IAM role - no API keys stored anywhere.
 */
export class CareSignalStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, "Vpc", { maxAzs: 2, natGateways: 1 });
    const cluster = new ecs.Cluster(this, "Cluster", { vpc, containerInsightsV2: ecs.ContainerInsights.ENABLED });

    const service = new ecsPatterns.ApplicationLoadBalancedFargateService(this, "Service", {
      cluster,
      cpu: 1024,
      memoryLimitMiB: 4096,
      desiredCount: 1,
      minHealthyPercent: 100,
      // Tasks stay in private subnets; only the load balancer is public.
      taskSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      publicLoadBalancer: true,
      circuitBreaker: { rollback: true },
      taskImageOptions: {
        image: ecs.ContainerImage.fromAsset(path.join(__dirname, "..", "..")),
        containerPort: 8000,
        environment: {
          CARESIGNAL_PROVIDER: "bedrock",
          AWS_REGION: this.region,
        },
        logDriver: ecs.LogDrivers.awsLogs({
          streamPrefix: "caresignal",
          logRetention: logs.RetentionDays.ONE_MONTH,
        }),
      },
    });

    service.targetGroup.configureHealthCheck({ path: "/api/health", healthyHttpCodes: "200" });
    // The first request after a deploy waits for model load; give the agent's
    // multi-tool turns room before the load balancer gives up.
    service.loadBalancer.setAttribute("idle_timeout.timeout_seconds", "300");

    service.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        resources: [
          `arn:${this.partition}:bedrock:*::foundation-model/anthropic.*`,
          `arn:${this.partition}:bedrock:*:${this.account}:inference-profile/*`,
        ],
      }),
    );

    new cdk.CfnOutput(this, "Url", { value: `http://${service.loadBalancer.loadBalancerDnsName}` });
  }
}

#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { CareSignalStack } from "../lib/caresignal-stack";

const app = new cdk.App();
new CareSignalStack(app, "CareSignal", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

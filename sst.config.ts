/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "chat-interface",
      removal: input?.stage === "production" ? "retain" : "remove",
      home: "aws",
    };
  },
    async run() {
      const vpc = new sst.aws.Vpc("HospitalNavigationInterfaceVpc", { bastion: true });
      const cluster = new sst.aws.Cluster("HospitalNavigationInterfaceCluster", { vpc });

      cluster.addService("ChatInterface", {
          cpu:  "1 vCPU",
          memory: "2 GB",
          loadBalancer: {
              domain: "api.yourdoc.click",
              ports: [
                  { listen: "80/http", redirect: "443/https" },
                  { listen: "443/https", forward: "80/http" }
              ],
          },
          dev: {
              command: "fastapi dev app/main.py",
          },
      });
  },
});

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

      cluster.addService("HChatInterface", {
          // cpu:  "0.5 vCPU",
          // memory: "1 GB",
          loadBalancer: {
              domain: "api.yourdoc.click",
              ports: [
                  { listen: "80/http", forward: "80/http" },
                  { listen: "443/https", forward: "80/http" }
              ],
          },
          dev: {
              command: "fastapi dev app/main.py",
          },
      });
  },
});

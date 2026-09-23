# k8s (phase 2)

k3d cluster: gateway + router + mock workers as Deployments/Services, HPA on
custom metrics, fake accelerator node labels (accelerator=nvidia-h100 | l40s
| apple-m5). Real Metal workers stay native on macOS, exposed via host
endpoints (host.docker.internal / NodeExtraCIDRs).

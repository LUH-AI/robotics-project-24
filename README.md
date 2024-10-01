# Home of Heinrich

[![Doc Status](https://github.com/LUH-AI/heinrich_template/actions/workflows/docs.yaml/badge.svg)](https://github.com/LUH-AI/heinrich_template/actions/workflows/docs.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Starter code for Heinrich the dog, including all the resources needed for getting started, basic usage and RL deployment.
Documentation: https://luh-ai.github.io/heinrich_template/

## Connecting to Heinrich

There are two primary ways of connecting to Heinrich: directly via SSH or via the AI net. The former is meant for deployment while the later is useful for package maintenance and updating.
For the purposes of deployment, you'll want to connect Heinrich to your computer with an ethernet cable, manually set the IPv4 adress to 192.168.123.51 and network mask to 255.255.255.0 then run:

```bash
make ping
```

If you get a positive answer, you can connect via:
```bash
make ssh
```
The default password is 123.

## Building the Backend

If you simply pull, you should have all executables present. If you can't execute them, you likely need to re-build the backend. This is only possible on aarch64 and x86_64 architectures currently - notably not on Apple computers with ARM chips!
We bundled all build commands in the makefile such that you only need to run:

```bash
make build-backend
```

Note that you need root rights for this at multiple points during the process!

## Running Deployment

For deployment, you'll need to start LCM and in a separate terminal run the actual deployment code. First test if LCM is connecting to the Unitree SDK:

```bash
make test-lcm
```

If that looks alright, you can go ahead and start LCM proper:

```bash
make start-lcm
```
This assumes eth0 is the correct interface for your robot. If it isn't this command will fail but print instructions of how to get the correct one.

Once LCM is listening, open a separate terminal. Here you can now start your deployment code, e.g. the Teddy Liao's example policy by running:
```bash
make deploy-example
```

For your own policies, you can put them into 'example_policies' and then run:
```bash
python deploy_policy.py
```
TODO: policy path should be an argumen
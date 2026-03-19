# Deploy to nodes without expect

Run these from your **Mac** in Terminal (from the project root). You'll be prompted for the node password each time.

---

## Node .100 (Pilot – cooledai-srv)

**1. Copy files** (password when prompted)

```bash
cd /Users/denniswork/Desktop/coolingai_simulator

scp -o StrictHostKeyChecking=no \
  scripts/st550_telemetry.py scripts/start_telemetry.sh \
  cooledaiadmin@192.168.12.100:~/

scp -o StrictHostKeyChecking=no \
  scripts/llama_workload_scheduler.py scripts/predictive_engine.py scripts/cooledai_agent.py \
  cooledaiadmin@192.168.12.100:~/
```

**2. SSH in with a TTY and run** (sudo will ask for password when needed)

```bash
ssh -t -o StrictHostKeyChecking=no cooledaiadmin@192.168.12.100
```

Then on the node (sudo may ask for password for start_telemetry and again for the agent):

```bash
sudo bash ~/start_telemetry.sh --node-id ST550-CooledAI-Predictive
pkill -f llama_workload_scheduler.py 2>/dev/null || true
pkill -f predictive_engine.py 2>/dev/null || true
pkill -f cooledai_agent.py 2>/dev/null || true
nohup python3 ~/predictive_engine.py > /tmp/cooledai_predictive_engine.log 2>&1 &
nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 &
sudo apt-get install -y ipmitool 2>/dev/null
sudo nohup python3 ~/cooledai_agent.py --api-url https://proactive-creativity-production.up.railway.app --api-key sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo --node-id ST550-CooledAI-Predictive --ipmi-variant lenovo >> /tmp/cooledai_agent.log 2>&1 &
exit
```

**Alternative:** SSH in and run by hand (no one-liner):

```bash
ssh -t -o StrictHostKeyChecking=no cooledaiadmin@192.168.12.100
```

Then on the node:

```bash
sudo bash ~/start_telemetry.sh --node-id ST550-CooledAI-Predictive
pkill -f llama_workload_scheduler.py 2>/dev/null || true; pkill -f predictive_engine.py 2>/dev/null || true
nohup python3 ~/predictive_engine.py > /tmp/cooledai_predictive_engine.log 2>&1 &
nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 &
exit
```

---

## Node .101 (Control – cooledai-control)

**1. Copy files** (password: `Admin123` when prompted)

```bash
cd /Users/denniswork/Desktop/coolingai_simulator

scp -o StrictHostKeyChecking=no \
  scripts/st550_telemetry.py scripts/start_telemetry.sh \
  scripts/llama_workload_scheduler.py \
  cooledaiadmin@192.168.12.101:~/
```

**2. SSH in with a TTY and run** (password: `Admin123`; sudo will ask again — use `-t`)

```bash
ssh -t -o StrictHostKeyChecking=no cooledaiadmin@192.168.12.101 'sudo bash ~/start_telemetry.sh --node-id ST550-Control-Traditional; pkill -f llama_workload_scheduler.py 2>/dev/null || true; nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>\&1 \&'
```

When prompted for **sudo** password, type `Admin123`.

**Alternative:** SSH in and run by hand:

```bash
ssh -t -o StrictHostKeyChecking=no cooledaiadmin@192.168.12.101
```

Then on the node:

```bash
sudo bash ~/start_telemetry.sh --node-id ST550-Control-Traditional
pkill -f llama_workload_scheduler.py 2>/dev/null || true
nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 &
exit
```

---

Done. Both nodes will have the new scheduler and telemetry running.

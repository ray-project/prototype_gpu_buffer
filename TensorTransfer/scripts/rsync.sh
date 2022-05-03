#!/bin/bash
echo "Using configure file $1 ..."
head_ip=$(ray get-head-ip $1)
worker_ips=$(ray get-worker-ips $1)
region=$(grep 'region:' $1 | sed 's/region: //g' | tr -d '[:space:]')
pem_file=~/.ssh/ray-autoscaler_$region.pem

echo "Using pemfile file $pem_file ..."

echo "copying to ubuntu@$head_ip"
rsync -avr --progress -e "ssh -o StrictHostKeyChecking=no -i $pem_file" "$2" "ubuntu@$head_ip:$3"

for worker_ip in $worker_ips; do
  echo "copying to ubuntu@$worker_ip"
  rsync -avr --progress -e "ssh -o StrictHostKeyChecking=no -i $pem_file" "$2" "ubuntu@$worker_ip:$3"
done
# services/airflow/dags/pipeline_mnist.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

DEFAULT_ARGS = {
    "owner": "you",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

# Set schedule to every 5 minutes
with DAG(
    dag_id="mnist_pipeline_5min",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
) as dag:

    # split raw -> processed
    split = BashOperator(
        task_id="split_data",
        bash_command="cd /opt/airflow/dags/../../.. && python code/datasets/split_data.py"
    )

    # train model
    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/dags/../../.. && python code/models/train_digits.py --epochs 5 --batch_size 64"
    )

    # deploy (bring up docker-compose services)
    deploy = BashOperator(
        task_id="deploy_services",
        bash_command="cd /opt/airflow/dags/../../.. && docker-compose -f code/deployment/docker-compose.yml up -d --build"
    )

    split >> train >> deploy

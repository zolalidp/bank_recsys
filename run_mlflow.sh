SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# .env не source-ится автоматически — без этого скрипт падает с "unbound
# variable" при запуске в свежей оболочке, где переменные ещё не экспортированы.
if [ -f "$SCRIPT_DIR/.env" ]; then
	set -a
	# shellcheck disable=SC1091
	source "$SCRIPT_DIR/.env"
	set +a
fi

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_BUCKET_NAME=$S3_BUCKET_NAME

# mlflow может быть не на PATH в свежей оболочке (venv не активирован).
if ! command -v mlflow >/dev/null 2>&1 && [ -x "$SCRIPT_DIR/.venv/bin/mlflow" ]; then
	PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi

mlflow server \
	--registry-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME \
	--backend-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME \
	--default-artifact-root s3://$AWS_BUCKET_NAME \
	--no-serve-artifacts
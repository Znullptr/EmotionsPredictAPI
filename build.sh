# install dependencies
pip install -r requirements.txt

# collect staticfiles

python manage.py collectstatic --no-input

# make migrations

python manage.py migrate
# Generated by Django 3.0.4 on 2020-05-06 11:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='crop',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cropname', models.CharField(max_length=100)),
                ('weather', models.CharField(max_length=50)),
            ],
        ),
    ]
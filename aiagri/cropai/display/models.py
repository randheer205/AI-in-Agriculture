from django.db import models

# Create your models here.
class crop(models.Model):
    cityname=models.CharField(max_length=100)
    temperature=models.FloatField()
    humidity=models.FloatField()
    ph=models.FloatField()
    rainfall=models.FloatField()
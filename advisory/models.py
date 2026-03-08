from django.db import models

class FarmInput(models.Model):
    DISTRICT_CHOICES = [
        ('angul', 'Angul'), ('balangir', 'Balangir'), ('balasore', 'Balasore'),
        ('bargarh', 'Bargarh'), ('bhadrak', 'Bhadrak'), ('boudh', 'Boudh'),
        ('cuttack', 'Cuttack'), ('deogarh', 'Deogarh'), ('dhenkanal', 'Dhenkanal'),
        ('gajapati', 'Gajapati'), ('ganjam', 'Ganjam'), ('jagatsinghpur', 'Jagatsinghpur'),
        ('jajpur', 'Jajpur'), ('jharsuguda', 'Jharsuguda'), ('kalahandi', 'Kalahandi'),
        ('kandhamal', 'Kandhamal'), ('kendrapara', 'Kendrapara'), ('keonjhar', 'Keonjhar'),
        ('khordha', 'Khordha'), ('koraput', 'Koraput'), ('malkangiri', 'Malkangiri'),
        ('mayurbhanj', 'Mayurbhanj'), ('nabarangpur', 'Nabarangpur'), ('nayagarh', 'Nayagarh'),
        ('nuapada', 'Nuapada'), ('puri', 'Puri'), ('rayagada', 'Rayagada'),
        ('sambalpur', 'Sambalpur'), ('sonepur', 'Sonepur'), ('sundargarh', 'Sundargarh')
    ]
    
    CROP_CHOICES = [
        ('rice', 'Rice'), ('maize', 'Maize'), ('wheat', 'Wheat'),
        ('groundnut', 'Groundnut'), ('mung', 'Mung'), ('cotton', 'Cotton'),
        ('sugarcane', 'Sugarcane'), ('turmeric', 'Turmeric')
    ]
    
    SEASON_CHOICES = [
        ('kharif', 'Kharif'), ('rabi', 'Rabi'), ('zaid', 'Zaid')
    ]
    
    IRRIGATION_CHOICES = [
        ('none', 'None/Rainfed'), ('tubewell', 'Tube well'),
        ('canal', 'Canal'), ('lift', 'Lift'), ('drip', 'Drip')
    ]
    
    SOIL_CHOICES = [
        ('alluvial', 'Alluvial'), ('lateritic', 'Lateritic'),
        ('red_black', 'Red & Black'), ('saline', 'Saline')
    ]
    
    SEED_CHOICES = [
        ('local', 'Local'), ('hyv', 'HYV'), ('hybrid', 'Hybrid')
    ]
    
    district = models.CharField(max_length=50, choices=DISTRICT_CHOICES)
    crop = models.CharField(max_length=50, choices=CROP_CHOICES)
    season = models.CharField(max_length=20, choices=SEASON_CHOICES)
    sowing_date = models.DateField()
    field_area = models.FloatField(help_text="Area in hectares")
    irrigation = models.CharField(max_length=20, choices=IRRIGATION_CHOICES)
    soil_type = models.CharField(max_length=20, choices=SOIL_CHOICES)
    soil_health_card = models.BooleanField(default=False)
    seed_variety = models.CharField(max_length=20, choices=SEED_CHOICES)
    pest_presence = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.crop} - {self.district} - {self.season}"

class Recommendation(models.Model):
    farm_input = models.OneToOneField(FarmInput, on_delete=models.CASCADE)
    predicted_yield = models.FloatField()
    confidence_interval = models.CharField(max_length=50)
    estimated_gain = models.FloatField()
    action_1 = models.TextField()
    action_2 = models.TextField()
    action_3 = models.TextField()
    reasoning = models.TextField()
    crop_suitability = models.TextField(blank=True, null=True)
    special_advice = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Recommendation for {self.farm_input}"

class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Contact from {self.name}: {self.subject}"

class PesticideShop(models.Model):
    DISTRICT_CHOICES = [
        ('angul', 'Angul'), ('balangir', 'Balangir'), ('balasore', 'Balasore'),
        ('bargarh', 'Bargarh'), ('bhadrak', 'Bhadrak'), ('boudh', 'Boudh'),
        ('cuttack', 'Cuttack'), ('deogarh', 'Deogarh'), ('dhenkanal', 'Dhenkanal'),
        ('gajapati', 'Gajapati'), ('ganjam', 'Ganjam'), ('jagatsinghpur', 'Jagatsinghpur'),
        ('jajpur', 'Jajpur'), ('jharsuguda', 'Jharsuguda'), ('kalahandi', 'Kalahandi'),
        ('kandhamal', 'Kandhamal'), ('kendrapara', 'Kendrapara'), ('keonjhar', 'Keonjhar'),
        ('khordha', 'Khordha'), ('koraput', 'Koraput'), ('malkangiri', 'Malkangiri'),
        ('mayurbhanj', 'Mayurbhanj'), ('nabarangpur', 'Nabarangpur'), ('nayagarh', 'Nayagarh'),
        ('nuapada', 'Nuapada'), ('puri', 'Puri'), ('rayagada', 'Rayagada'),
        ('sambalpur', 'Sambalpur'), ('sonepur', 'Sonepur'), ('sundargarh', 'Sundargarh')
    ]
    
    shop_name = models.CharField(max_length=200)
    owner_name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15)
    email = models.EmailField(blank=True, null=True)
    district = models.CharField(max_length=50, choices=DISTRICT_CHOICES)
    address = models.TextField()
    pincode = models.CharField(max_length=10)
    pesticides_available = models.TextField(help_text="List of pesticides available (comma-separated)")
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.shop_name} - {self.get_district_display()}"

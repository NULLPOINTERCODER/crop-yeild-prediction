from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.http import JsonResponse
from django.conf import settings
from .forms import FarmInputForm, SignupForm, ContactForm, PesticideShopForm
from .models import FarmInput, Recommendation, Contact, PesticideShop
from .ml_model import yield_predictor
import traceback
import requests
import openai
import os
from datetime import datetime

def home(request):
    """Home page view"""
    return render(request, 'advisory/home.html')

@login_required(login_url='/login/')
def farm_input(request):
    """Farm input form view"""
    if request.method == 'POST':
        form = FarmInputForm(request.POST)
        if form.is_valid():
            try:
                farm_input_obj = form.save()
                
                # Generate ML prediction
                predicted_yield, confidence = yield_predictor.predict_yield(farm_input_obj)
                print(f"DEBUG: Predicted yield: {predicted_yield}")
                
                recommendations = yield_predictor.generate_recommendations(farm_input_obj, predicted_yield)
                print(f"DEBUG: AI Recommendations generated: {recommendations}")
                
                # Ensure all required fields are present
                if not all(key in recommendations for key in ['action_1', 'action_2', 'action_3', 'reasoning', 'estimated_gain', 'crop_suitability', 'special_advice']):
                    raise ValueError("Incomplete recommendation data generated")
                
                # Save recommendation
                recommendation = Recommendation.objects.create(
                    farm_input=farm_input_obj,
                    predicted_yield=float(predicted_yield),
                    confidence_interval=str(confidence),
                    estimated_gain=float(recommendations['estimated_gain']),
                    action_1=str(recommendations['action_1']),
                    action_2=str(recommendations['action_2']),
                    action_3=str(recommendations['action_3']),
                    reasoning=str(recommendations['reasoning']),
                    crop_suitability=str(recommendations.get('crop_suitability', '')),
                    special_advice=str(recommendations.get('special_advice', ''))
                )
                
                messages.success(request, "AI recommendation generated successfully!")
                return redirect('recommendation', recommendation_id=recommendation.id)
                
            except Exception as e:
                messages.error(request, f"Error generating recommendation: {str(e)}. Please try again.")
                print(f"Error in farm_input view: {e}")  # For debugging
        else:
            messages.error(request, "Please correct the errors in the form.")
                
    else:
        form = FarmInputForm()
    
    return render(request, 'advisory/farm_input.html', {'form': form})

@login_required(login_url='/login/')
def recommendation(request, recommendation_id):
    """Display recommendation results"""
    try:
        recommendation = Recommendation.objects.get(id=recommendation_id)
        
        # Get district average for comparison
        farm_input = recommendation.farm_input
        district_avg = yield_predictor.get_district_average(farm_input.district, farm_input.crop, farm_input.season)
        
        # Calculate total production
        total_production = recommendation.predicted_yield * farm_input.field_area
        
        # Calculate improvement percentage
        improvement = 0
        if district_avg > 0:
            improvement = ((recommendation.predicted_yield - district_avg) / district_avg) * 100
        
        context = {
            'recommendation': recommendation,
            'total_production': total_production,
            'yield_comparison': {
                'predicted': recommendation.predicted_yield,
                'district_avg': district_avg,
                'improvement': improvement
            }
        }
        
        return render(request, 'advisory/recommendation.html', context)
    except Recommendation.DoesNotExist:
        messages.error(request, "Recommendation not found. Please try generating a new recommendation.")
        return redirect('farm_input')
    except Exception as e:
        messages.error(request, f"Error loading recommendation: {str(e)}")
        return redirect('farm_input')

def about(request):
    """About page view"""
    return render(request, 'advisory/about.html')

# from django.conf import settings
# from twilio.rest import Client
# import logging

from twilio.rest import Client
import logging

def contact(request):
    """Contact form view"""
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            contact_instance = form.save()
            # Send SMS notification via Twilio
            try:
                client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                message_body = (
                    f"New contact form submission:\n"
                    f"Name: {contact_instance.name}\n"
                    f"Email: {contact_instance.email}\n"
                    f"Subject: {contact_instance.subject}\n"
                    f"Message: {contact_instance.message}"
                )
                client.messages.create(
                    body=message_body,
                    from_=settings.TWILIO_PHONE_NUMBER,
                    to=settings.DEVELOPER_MOBILE_NUMBER
                )
            except Exception as e:
                logging.error(f"Failed to send SMS notification: {e}")
            messages.success(request, "Thank you for your message! We'll get back to you soon.")
            return redirect('home')
        else:
            messages.error(request, "Please correct the errors in the form.")
    else:
        form = ContactForm()
    return render(request, 'advisory/contact.html', {'form': form})

def signup(request):
    """User signup view"""
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created successfully! Welcome to Krishi Salahkar.")
            return redirect('home')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = SignupForm()
    return render(request, 'advisory/signup.html', {'form': form})

def weather_forecast(request):
    """Weather forecast view"""
    location = request.GET.get('location', 'Bhubaneswar')
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    api_key = settings.WEATHER_API_KEY
    base_url = settings.WEATHER_API_BASE_URL

    if api_key == 'your-weather-api-key-here':
        messages.error(request, "Weather API key not configured. Please set WEATHER_API_KEY in settings.")
        return render(request, 'advisory/weather.html', {'error': 'API key not configured'})

    try:
        # Use coordinates if available, otherwise use city name
        if lat and lon:
            current_url = f"{base_url}/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            forecast_url = f"{base_url}/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        else:
            current_url = f"{base_url}/weather?q={location}&appid={api_key}&units=metric"
            forecast_url = f"{base_url}/forecast?q={location}&appid={api_key}&units=metric"
        
        current_response = requests.get(current_url)
        current_data = current_response.json()

        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()

        if current_response.status_code != 200 or forecast_response.status_code != 200:
            error_msg = f"API Error - Current: {current_response.status_code} ({current_response.text}), Forecast: {forecast_response.status_code} ({forecast_response.text})"
            messages.error(request, error_msg)
            return render(request, 'advisory/weather.html', {'error': error_msg})

        # Process forecast data (group by day)
        daily_forecasts = {}
        hourly_forecasts = []
        for item in forecast_data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily_forecasts:
                daily_forecasts[date] = {
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon'],
                    'wind_speed': item['wind']['speed'],
                    'date': date
                }
            else:
                daily_forecasts[date]['temp_min'] = min(daily_forecasts[date]['temp_min'], item['main']['temp_min'])
                daily_forecasts[date]['temp_max'] = max(daily_forecasts[date]['temp_max'], item['main']['temp_max'])

            # Collect hourly forecasts (next 24 hours, 8 entries since 3-hour intervals)
            if len(hourly_forecasts) < 8:
                hourly_forecasts.append({
                    'time': item['dt_txt'].split(' ')[1][:5],  # HH:MM
                    'temp': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon'],
                    'wind_speed': item['wind']['speed']
                })

        # Get current date and time
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M:%S")

        context = {
            'current_weather': current_data,
            'daily_forecasts': list(daily_forecasts.values())[:5],
            'hourly_forecasts': hourly_forecasts,
            'location': current_data.get('name', location),
            'current_date': current_date,
            'current_day': current_day,
            'current_time': current_time
        }
        return render(request, 'advisory/weather.html', context)

    except Exception as e:
        messages.error(request, f"Error fetching weather data: {str(e)}")
        return render(request, 'advisory/weather.html', {'error': str(e)})


from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def chatbot(request):
    """Chatbot API endpoint"""
    import logging
    if request.method == 'POST':
        try:
            user_message = request.POST.get('message', '').strip()

            if not user_message:
                return JsonResponse({'error': 'Message is required'}, status=400)

            # Set Together AI API key from settings

            # Improved farming-specific context with kindness and responsiveness emphasis
            system_prompt = """You are Krishi Salahkar, a kind, polite, and responsive AI agricultural assistant specializing in farming advice for Odisha, India.
            You help farmers with:
            - Crop selection and planning
            - Soil management and fertilization
            - Pest and disease control
            - Irrigation techniques
            - Weather-based farming decisions
            - Sustainable farming practices
            - Odisha-specific agricultural knowledge

            Always provide practical, actionable advice based on scientific farming principles.
            Keep responses concise, informative, and kind.
            Always respond in a friendly and encouraging manner.
            If you don't know something specific to Odisha, acknowledge this politely and provide general best practices."""

            # Create chat completion using Together AI API
            client = openai.OpenAI(api_key=settings.TOGETHER_AI_API_KEY, base_url="https://api.together.xyz/v1")
            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )

            bot_response = response.choices[0].message.content.strip()
            logging.info(f"User message: {user_message}")
            logging.info(f"Bot response: {bot_response}")

            return JsonResponse({
                'response': bot_response,
                'status': 'success'
            })

        except Exception as e:
            import traceback
            traceback.print_exc()  # Print full traceback
            logging.error(f"Chatbot error: {str(e)}")

            # Check if error is related to Together AI API key or request
            if 'Invalid' in str(e) or 'Unauthorized' in str(e) or 'api key' in str(e).lower():
                error_message = "Together AI API key error: Please check your API key configuration."
                logging.error(error_message)
                return JsonResponse({'error': error_message}, status=500)

            # Expanded fallback responses for common farming questions
            fallback_responses = {
                'rice': 'For rice cultivation in Odisha, use varieties like Naveen or Swarna. Plant during Kharif season (June-July) with proper irrigation and apply NPK fertilizers.',
                'wheat': 'Wheat grows well in Odisha during Rabi season. Use varieties like HD-2967. Ensure proper irrigation and apply fertilizers at recommended doses.',
                'maize': 'Maize is a good crop for Odisha. Plant in well-drained soil and use recommended fertilizers for best yield.',
                'pest': 'For pest control, use integrated pest management techniques including biological controls and safe pesticides.',
                'default': 'I\'m here to help with your farming questions. Please ask about crops, irrigation, fertilizers, or any agricultural topic! I\'ll do my best to assist you kindly.'
            }

            user_lower = user_message.lower()
            fallback_response = fallback_responses['default']

            for key, response in fallback_responses.items():
                if key in user_lower and key != 'default':
                    fallback_response = response
                    break

            return JsonResponse({
                'response': fallback_response,
                'status': 'success'
            })

    return JsonResponse({'error': 'Method not allowed'}, status=405)

def register_shop(request):
    """Pesticide shop registration view"""
    if request.method == 'POST':
        form = PesticideShopForm(request.POST)
        if form.is_valid():
            shop = form.save()
            messages.success(request, "Shop registered successfully! It will be verified and listed soon.")
            return redirect('home')
        else:
            messages.error(request, "Please correct the errors in the form.")
    else:
        form = PesticideShopForm()
    return render(request, 'advisory/register_shop.html', {'form': form})

def nearby_shops(request, district=None):
    """Find nearby pesticide shops in a district or by coordinates"""
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    
    if lat and lon:
        # Calculate distance for all shops and sort by nearest
        from math import radians, sin, cos, sqrt, atan2
        
        user_lat = float(lat)
        user_lon = float(lon)
        
        shops = PesticideShop.objects.all()
        shops_with_distance = []
        
        # Approximate coordinates for Odisha districts (center points)
        district_coords = {
            'angul': (20.8400, 85.1000), 'balangir': (20.7100, 83.4900),
            'balasore': (21.4934, 86.9336), 'bargarh': (21.3333, 83.6167),
            'bhadrak': (21.0542, 86.4916), 'boudh': (20.8350, 84.3300),
            'cuttack': (20.4625, 85.8830), 'deogarh': (21.5333, 84.7333),
            'dhenkanal': (20.6667, 85.6000), 'gajapati': (18.8667, 84.1333),
            'ganjam': (19.3850, 84.8000), 'jagatsinghpur': (20.2667, 86.1667),
            'jajpur': (20.8500, 86.3333), 'jharsuguda': (21.8533, 84.0067),
            'kalahandi': (19.9100, 83.1650), 'kandhamal': (20.1667, 84.1000),
            'kendrapara': (20.5000, 86.4200), 'keonjhar': (21.6300, 85.5833),
            'khordha': (20.1809, 85.6100), 'koraput': (18.8100, 82.7100),
            'malkangiri': (18.3667, 81.8833), 'mayurbhanj': (21.9333, 86.7333),
            'nabarangpur': (19.2333, 82.5500), 'nayagarh': (20.1300, 85.0967),
            'nuapada': (20.7667, 82.6167), 'puri': (19.8106, 85.8314),
            'rayagada': (19.1667, 83.4167), 'sambalpur': (21.4704, 83.9701),
            'sonepur': (20.8333, 83.9167), 'sundargarh': (22.1167, 84.0333)
        }
        
        for shop in shops:
            if shop.district in district_coords:
                shop_lat, shop_lon = district_coords[shop.district]
                
                # Haversine formula to calculate distance
                R = 6371  # Earth radius in km
                lat1, lon1 = radians(user_lat), radians(user_lon)
                lat2, lon2 = radians(shop_lat), radians(shop_lon)
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = R * c
                
                shops_with_distance.append({
                    'shop': shop,
                    'distance': round(distance, 1)
                })
        
        # Sort by distance
        shops_with_distance.sort(key=lambda x: x['distance'])
        
        return render(request, 'advisory/nearby_shops.html', {
            'shops_with_distance': shops_with_distance,
            'district': 'your location',
            'by_location': True
        })
    
    elif district:
        shops = PesticideShop.objects.filter(district=district)
        return render(request, 'advisory/nearby_shops.html', {
            'shops': shops,
            'district': district,
            'by_location': False
        })
    else:
        shops = PesticideShop.objects.all()
        return render(request, 'advisory/nearby_shops.html', {
            'shops': shops,
            'district': 'all areas',
            'by_location': False
        })

import random
import os
from django.conf import settings
import statistics
import pickle
import openai

class YieldPredictor:
    def __init__(self):
        self.data = None
        self.is_loaded = False
        self.model = None
        self.load_model()
        
    def load_data(self):
        """Load and preprocess the combined_tables data"""
        if self.is_loaded and self.data:
            return self.data
            
        try:
            data_path = os.path.join(settings.BASE_DIR, 'combined_tables.txt')
            data = []
            with open(data_path, 'r') as f:
                lines = f.readlines()
                headers = lines[0].strip().split('\t')
                for line in lines[1:]:
                    values = line.strip().split('\t')
                    if len(values) == len(headers):
                        row = dict(zip(headers, values))
                        # Convert numeric fields
                        try:
                            row['yield'] = float(row['yield'])
                            row['field_area'] = float(row['field_area'])
                            row['rainfall'] = float(row['rainfall'])
                            row['year'] = int(row['year'])
                        except:
                            continue
                        data.append(row)
            self.data = data
            self.is_loaded = True
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def load_model(self):
        """Load the trained model from pickle file"""
        model_path = os.path.join(settings.BASE_DIR, 'advisory/models/farm_model.pkl')
        try:
            from sklearn.ensemble import RandomForestRegressor
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✓ Trained ML model loaded successfully")
        except Exception as e:
            print(f"Model not available, using rule-based prediction: {e}")
            self.model = None

    def prepare_features(self, farm_input):
        """Prepare numerical features for the model"""
        # Encode categorical variables
        crop_map = {'rice': 1, 'maize': 2, 'wheat': 3, 'groundnut': 4, 'mung': 5, 'cotton': 6, 'sugarcane': 7, 'turmeric': 8}
        district_map = {
            'angul': 1, 'balangir': 2, 'balasore': 3, 'bargarh': 4, 'bhadrak': 5, 'boudh': 6,
            'cuttack': 7, 'deogarh': 8, 'dhenkanal': 9, 'gajapati': 10, 'ganjam': 11, 'jagatsinghpur': 12,
            'jajpur': 13, 'jharsuguda': 14, 'kalahandi': 15, 'kandhamal': 16, 'kendrapara': 17, 'keonjhar': 18,
            'khordha': 19, 'koraput': 20, 'malkangiri': 21, 'mayurbhanj': 22, 'nabarangpur': 23, 'nayagarh': 24,
            'nuapada': 25, 'puri': 26, 'rayagada': 27, 'sambalpur': 28, 'sonepur': 29, 'sundargarh': 30
        }
        season_map = {'kharif': 1, 'rabi': 2, 'zaid': 3}
        irrigation_map = {'none': 0, 'drip': 1, 'tubewell': 2, 'canal': 3, 'lift': 4}
        seed_map = {'local': 0, 'hyv': 1, 'hybrid': 2}
        soil_map = {'alluvial': 1, 'red_black': 2, 'lateritic': 3, 'saline': 4}

        features = [
            crop_map.get(farm_input.crop, 0),
            district_map.get(farm_input.district, 0),
            season_map.get(farm_input.season, 0),
            irrigation_map.get(farm_input.irrigation, 0),
            seed_map.get(farm_input.seed_variety, 0),
            soil_map.get(farm_input.soil_type, 0),
            1 if farm_input.soil_health_card else 0,
            1 if farm_input.pest_presence else 0
        ]
        return features

    def predict_yield(self, farm_input):
        """Predict yield based on farm input using the trained model or rule-based fallback"""
        if self.model:
            try:
                features = self.prepare_features(farm_input)
                prediction = self.model.predict([features])[0]
                # Ensure positive prediction
                prediction = max(prediction, 100)
                # Calculate confidence interval (simplified)
                confidence_range = prediction * 0.12
                confidence = f"±{confidence_range:.0f}"
                return prediction, confidence
            except Exception as e:
                print(f"Model prediction failed: {e}, falling back to rule-based")

        # Fallback to rule-based prediction
        data = self.load_data()

        # Base yields for different crops (kg/ha)
        base_yields = {
            'rice': 3200, 'maize': 4200, 'wheat': 3500, 'groundnut': 2200,
            'mung': 1100, 'cotton': 1600, 'sugarcane': 75000, 'turmeric': 5200
        }

        base_yield = base_yields.get(farm_input.crop, 2500)

        # Apply adjustments based on farming practices
        multiplier = 1.0

        # Irrigation adjustment
        if farm_input.irrigation == 'drip':
            multiplier *= 1.25
        elif farm_input.irrigation in ['tubewell', 'canal']:
            multiplier *= 1.15
        elif farm_input.irrigation == 'lift':
            multiplier *= 1.08
        elif farm_input.irrigation == 'none':
            multiplier *= 0.85

        # Seed variety adjustment
        if farm_input.seed_variety == 'hybrid':
            multiplier *= 1.20
        elif farm_input.seed_variety == 'hyv':
            multiplier *= 1.10
        elif farm_input.seed_variety == 'local':
            multiplier *= 0.95

        # Soil type adjustment
        if farm_input.soil_type == 'alluvial':
            multiplier *= 1.05
        elif farm_input.soil_type == 'red_black':
            multiplier *= 1.02
        elif farm_input.soil_type == 'lateritic':
            multiplier *= 0.98
        elif farm_input.soil_type == 'saline':
            multiplier *= 0.85

        # Season adjustment
        if farm_input.season == 'kharif':
            multiplier *= 1.05  # Monsoon advantage
        elif farm_input.season == 'rabi':
            multiplier *= 1.10  # Better conditions
        elif farm_input.season == 'zaid':
            multiplier *= 0.95  # Summer challenges

        # Soil health card bonus
        if farm_input.soil_health_card:
            multiplier *= 1.05

        # Pest presence penalty
        if farm_input.pest_presence:
            multiplier *= 0.92

        # Calculate final prediction
        prediction = base_yield * multiplier

        # Add some randomness for realism
        variation = random.uniform(0.95, 1.05)
        prediction *= variation

        # Calculate confidence interval
        confidence_range = prediction * 0.12
        confidence = f"±{confidence_range:.0f}"

        return max(prediction, 100), confidence
    
    def generate_recommendations(self, farm_input, predicted_yield):
        """Generate AI-powered actionable recommendations"""
        import openai
        from django.conf import settings
        
        try:
            # Create prompt for AI
            prompt = f"""You are an expert agricultural advisor for Odisha, India. Analyze this farm data and provide recommendations.

Farm Details:
- Crop: {farm_input.get_crop_display()}
- District: {farm_input.get_district_display()}
- Season: {farm_input.get_season_display()}
- Field Area: {farm_input.field_area} hectares
- Irrigation: {farm_input.get_irrigation_display()}
- Seed Variety: {farm_input.get_seed_variety_display()}
- Soil Type: {farm_input.get_soil_type_display()}
- Soil Health Card: {'Yes' if farm_input.soil_health_card else 'No'}
- Pest/Disease Present: {'Yes' if farm_input.pest_presence else 'No'}
- Predicted Yield: {predicted_yield:.0f} kg/ha

Provide recommendations in this EXACT format:

ACTION 1: [specific action with timing]
ACTION 2: [specific action with timing]
ACTION 3: [specific action with timing]

CROP SUITABILITY: [Analyze if this crop is suitable for the given soil type, season, and irrigation. If NOT suitable, suggest 2-3 better alternative crops for these conditions]

SPECIAL ADVICE: [One important tip specific to this farm's conditions in Odisha]

Focus on: irrigation, fertilization, pest control, and yield optimization."""

            client = openai.OpenAI(api_key=settings.TOGETHER_AI_API_KEY, base_url="https://api.together.xyz/v1")
            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"AI Response: {ai_response}")
            
            # Parse AI response
            actions = []
            crop_suitability = ""
            special_advice = ""
            
            lines = ai_response.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('ACTION'):
                    action_text = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                    actions.append(action_text)
                elif 'CROP SUITABILITY' in line:
                    # Get content after this line
                    crop_suitability = '\n'.join(lines[i+1:]).split('SPECIAL ADVICE')[0].strip()
                elif 'SPECIAL ADVICE' in line:
                    special_advice = '\n'.join(lines[i+1:]).strip()
            
            # Ensure we have 3 actions
            while len(actions) < 3:
                actions.append("Monitor crop regularly and consult local agriculture officer")
            
            if not crop_suitability:
                crop_suitability = f"{farm_input.get_crop_display()} is suitable for {farm_input.get_soil_type_display()} soil in {farm_input.get_season_display()} season."
            
            if not special_advice:
                special_advice = "Maintain regular field monitoring and follow recommended agricultural practices."
            
            potential_gain = self._calculate_potential_gain(farm_input)
            
            return {
                'action_1': actions[0],
                'action_2': actions[1],
                'action_3': actions[2],
                'reasoning': f"AI-powered analysis for {farm_input.get_crop_display()} cultivation in {farm_input.get_district_display()} district.",
                'estimated_gain': potential_gain,
                'crop_suitability': crop_suitability,
                'special_advice': special_advice
            }
            
        except Exception as e:
            print(f"AI recommendation failed: {e}, using fallback")
            # Fallback to rule-based
            potential_gain = self._calculate_potential_gain(farm_input)
            return {
                'action_1': self._get_priority_action_1(farm_input),
                'action_2': self._get_priority_action_2(farm_input),
                'action_3': self._get_priority_action_3(farm_input),
                'reasoning': f"Analysis of {farm_input.get_crop_display()} in {farm_input.get_district_display()} during {farm_input.get_season_display()} season.",
                'estimated_gain': potential_gain,
                'crop_suitability': self._get_crop_suitability(farm_input),
                'special_advice': self._get_special_advice(farm_input)
            }
    
    def _calculate_potential_gain(self, farm_input):
        """Calculate potential yield gain from improved practices"""
        gain = 0
        
        # Irrigation improvement potential
        if farm_input.irrigation == 'none':
            gain += 20
        elif farm_input.irrigation in ['lift', 'canal']:
            gain += 8
        elif farm_input.irrigation == 'tubewell':
            gain += 5
        
        # Seed variety improvement
        if farm_input.seed_variety == 'local':
            gain += 15
        elif farm_input.seed_variety == 'hyv':
            gain += 8
        
        # Soil health improvement
        if not farm_input.soil_health_card:
            gain += 10
        
        # Pest management
        if farm_input.pest_presence:
            gain += 12
        
        # Cap the gain between 5-25%
        return min(max(gain, 5), 25)
    
    def _get_priority_action_1(self, farm_input):
        """Get highest priority immediate action"""
        if farm_input.irrigation == 'none' and farm_input.crop in ['rice', 'sugarcane']:
            return "URGENT: Install irrigation system immediately - these crops need consistent water supply for survival and yield"
        elif farm_input.pest_presence:
            return "IMMEDIATE: Apply integrated pest management - spray neem oil and set up pheromone traps within 2 days"
        elif not farm_input.soil_health_card:
            return "HIGH PRIORITY: Get soil health card from nearest agriculture office to optimize fertilizer application"
        elif farm_input.irrigation == 'none':
            return "CRITICAL: Install drip irrigation system to increase water efficiency and boost yield by 20-25%"
        else:
            return "OPTIMIZE: Monitor soil moisture daily and apply irrigation at critical crop growth stages"
    
    def _get_priority_action_2(self, farm_input):
        """Get second priority action"""
        if farm_input.soil_health_card:
            crop_fertilizer = {
                'rice': "Apply 120:60:40 NPK kg/ha in 3 splits - 50% basal, 25% tillering, 25% panicle stage",
                'maize': "Apply 150:75:40 NPK kg/ha - 1/3 at sowing, 1/3 at knee-high, 1/3 at tasseling",
                'wheat': "Apply 120:60:40 NPK kg/ha in 3 splits based on soil test recommendations",
                'groundnut': "Apply 20:60:40 NPK kg/ha - groundnut fixes nitrogen naturally",
                'cotton': "Apply 150:75:75 NPK kg/ha in splits with micronutrients",
                'sugarcane': "Apply 300:150:150 NPK kg/ha in 4 splits throughout growing season"
            }
            return crop_fertilizer.get(farm_input.crop, "Apply balanced NPK fertilizer as per soil test in 2-3 splits")
        else:
            return "Apply balanced NPK fertilizer (consult agriculture officer) and get soil testing done immediately"
    
    def _get_priority_action_3(self, farm_input):
        """Get third priority action"""
        if farm_input.seed_variety == 'local':
            return f"UPGRADE: Switch to hybrid or HYV {farm_input.crop} varieties for 15-20% higher yield next season"
        elif farm_input.season == 'kharif':
            return "WEATHER PREP: Monitor weather forecasts and provide drainage during heavy rains to prevent waterlogging"
        elif farm_input.season == 'rabi':
            return "TIMING: Ensure timely sowing and harvest to avoid heat stress and maximize market prices"
        else:
            return "FIELD MANAGEMENT: Maintain proper plant spacing, weed control, and regular field monitoring for diseases"

    def _get_special_advice(self, farm_input):
        """Get special advice for the farm"""
        return f"For {farm_input.get_crop_display()} in {farm_input.get_season_display()} season, ensure timely operations and monitor weather conditions regularly."
    
    def _get_crop_suitability(self, farm_input):
        """Analyze crop suitability and suggest alternatives if needed"""
        unsuitable_combinations = {
            ('rice', 'saline'): "Rice struggles in saline soil. Consider: Barley, Mustard, or Salt-tolerant rice varieties.",
            ('wheat', 'lateritic'): "Wheat is not ideal for lateritic soil. Better options: Millets, Pulses, or Groundnut.",
            ('sugarcane', 'none'): "Sugarcane needs irrigation. Without it, grow: Millets, Pulses, or Groundnut instead.",
            ('rice', 'zaid'): "Rice in summer (Zaid) is challenging. Consider: Mung, Watermelon, or Cucumber.",
        }
        
        key = (farm_input.crop, farm_input.soil_type)
        if key in unsuitable_combinations:
            return unsuitable_combinations[key]
        
        key = (farm_input.crop, farm_input.irrigation)
        if key in unsuitable_combinations:
            return unsuitable_combinations[key]
        
        key = (farm_input.crop, farm_input.season)
        if key in unsuitable_combinations:
            return unsuitable_combinations[key]
        
        return f"{farm_input.get_crop_display()} is suitable for your {farm_input.get_soil_type_display()} soil in {farm_input.get_season_display()} season with {farm_input.get_irrigation_display()} irrigation."

    def get_district_average(self, district, crop, season):
        """Get average yield for district, crop, season combination"""
        # District and crop specific averages (simplified)
        district_crop_avg = {
            'rice': 2800, 'maize': 3500, 'wheat': 3000, 'groundnut': 1900,
            'mung': 950, 'cotton': 1400, 'sugarcane': 68000, 'turmeric': 4500
        }
        
        base_avg = district_crop_avg.get(crop, 2200)
        
        # Season adjustment
        if season == 'kharif':
            base_avg *= 0.95
        elif season == 'rabi':
            base_avg *= 1.05
        elif season == 'zaid':
            base_avg *= 0.90
        
        return base_avg

# Global instance
yield_predictor = YieldPredictor()
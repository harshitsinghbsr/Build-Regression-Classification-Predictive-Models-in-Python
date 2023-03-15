import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# read the CSV file into a pandas DataFrame
disease = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/MultipleDiseasePrediction.csv")

# split the data into training and testing sets
X = disease[['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
             'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
             'burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets',
             'mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level',
             'cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache',
             'yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
             'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes',
             'acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise',
             'blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure',
             'runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
             'pain_in_anal_region','bloody_stool',
             'brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts',
             'drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness',
             'stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance',
             'unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
             'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression',
             'irritability','muscle_pain','altered_sensorium',
             'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
             'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
             'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
             'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
             'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
             'yellow_crust_ooze']]
y = disease ['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit a KNN model to the training data
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# make predictions on the test data
y_pred = classifier.predict(X_test)

# evaluate the model using a classification report
print(classification_report(y_test, y_pred))

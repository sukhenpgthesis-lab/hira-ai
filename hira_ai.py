import os
import json
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
from datetime import datetime
import pytz
import firebase_admin

from firebase_admin import credentials, db, messaging

# ==============================
# Firebase Setup
# ==============================
firebase_json = json.loads(os.environ["FIREBASE_JSON"])
cred = credentials.Certificate(firebase_json)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://hira-iot-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

# ==============================
# Timezone + Constants
# ==============================
ist = pytz.timezone("Asia/Kolkata")

CO2_EMISSION_FACTOR = 1.224
LOOP_INTERVAL_SEC = 20

# ==============================
# Load ANN Models
# ==============================
high_model = tf.keras.models.load_model("hira_high_power_model.keras")
high_scaler = joblib.load("hira_high_power_scaler.pkl")

low_model = tf.keras.models.load_model("hira_low_power_model.keras")
low_scaler = joblib.load("hira_low_power_scaler.pkl")

# ==============================
# ThingSpeak
# ==============================
CHANNEL_ID = "3339649"
READ_API_KEY = "LF573A0E1FH6MOCW"

# ==============================
# Status Function
# ==============================
def get_status(eff, mode):
    if mode == "LOW_POWER_ANN":
        return "Low Power Normal" if eff >= 80 else "Low Power Check"

    if eff >= 90:
        return "Healthy"
    elif eff >= 75:
        return "Moderate"
    elif eff >= 50:
        return "Low Efficiency"
    else:
        return "Check System"

# ==============================
# Fault Detection
# ==============================
def detect_fault(actual, predicted, efficiency, voltage, current, lux, ghi, relay_count):

    if predicted <= 0:
        return {
            "alarm_level": "ALARM",
            "fault_type": "Prediction Error",
            "fault_message": "ANN predicted invalid power",
            "soiling_loss": 0.0,
            "performance_ratio": 0.0,
            "power_loss": 0.0
        }

    power_loss = max(predicted - actual, 0)
    soiling_loss = float(np.clip((power_loss / predicted) * 100, 0, 100))
    performance_ratio = float(np.clip(actual / predicted, 0, 1.5))

    if lux > 100 and ghi > 1 and actual < 0.5:
        level = "CRITICAL"
        fault_type = "No Power Generation"
        fault_message = "Solar input available but output power is almost zero"

    elif voltage < 2 and lux > 100:
        level = "CRITICAL"
        fault_type = "Voltage Fault"
        fault_message = "Panel, buck converter, wiring, or load connection problem possible"

    elif relay_count > 0 and current < 0.01 and voltage > 2:
        level = "WARNING"
        fault_type = "Current Fault"
        fault_message = "Load is connected but current is very low"

    elif efficiency >= 85:
        level = "NORMAL"
        fault_type = "No Fault"
        fault_message = "System operating normally"

    elif efficiency >= 70:
        level = "WARNING"
        fault_type = "Minor Loss"
        fault_message = "Small efficiency drop detected"

    elif efficiency >= 50:
        level = "ALARM"
        fault_type = "Moderate Loss"
        fault_message = "Possible soiling, shading, weak irradiance, or load mismatch"

    else:
        level = "CRITICAL"
        fault_type = "Major Fault"
        fault_message = "High power loss detected. Check panel, wiring, relay load, and sensor values"

    return {
        "alarm_level": level,
        "fault_type": fault_type,
        "fault_message": fault_message,
        "soiling_loss": soiling_loss,
        "performance_ratio": performance_ratio,
        "power_loss": power_loss
    }

# ==============================
# Premium FCM Notification
# ==============================
def send_premium_notification(fault_data, now):
    message = messaging.Message(
        notification=messaging.Notification(
            title="🚨 HIRA Critical Energy Alert",
            body=f"{fault_data['fault_type']} • {fault_data['fault_message']}"
        ),
        data={
            "alarm_level": str(fault_data["alarm_level"]),
            "fault_type": str(fault_data["fault_type"]),
            "details": str(fault_data["fault_message"]),
            "time": str(now)
        },
        topic="hira_alerts",
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                channel_id="hira_critical_alerts",
                title="🚨 HIRA Critical Energy Alert",
                body=f"{fault_data['fault_type']}\n{fault_data['fault_message']}",
                icon="ic_launcher_foreground",
                color="#EF4444",
                sound="default",
                tag="hira_energy_alert"
            )
        )
    )

    response = messaging.send(message)
    print("Premium FCM Sent:", response)

# ==============================
# Energy Stats Function
# ==============================
def update_energy_stats(actual_power, now_dt, now):
    today_key = now_dt.strftime("%Y-%m-%d")
    month_key = now_dt.strftime("%Y-%m")

    energy_increment_kwh = (actual_power * LOOP_INTERVAL_SEC) / 3600.0 / 1000.0
    energy_increment_kwh = max(energy_increment_kwh, 0)

    co2_increment_kg = energy_increment_kwh * CO2_EMISSION_FACTOR

    daily_ref = db.reference("HIRA/energy_stats/daily").child(today_key)
    daily_old = daily_ref.get() or {}

    daily_energy_kwh = float(daily_old.get("energy_kwh", 0)) + energy_increment_kwh
    daily_co2_kg = float(daily_old.get("co2_saved_kg", 0)) + co2_increment_kg

    daily_ref.set({
        "date": today_key,
        "energy_kwh": round(daily_energy_kwh, 6),
        "co2_saved_kg": round(daily_co2_kg, 6),
        "co2_factor_kg_per_kwh": CO2_EMISSION_FACTOR,
        "last_increment_energy_kwh": round(energy_increment_kwh, 8),
        "last_increment_co2_kg": round(co2_increment_kg, 8),
        "last_updated": now
    })

    monthly_ref = db.reference("HIRA/energy_stats/monthly").child(month_key)
    monthly_old = monthly_ref.get() or {}

    monthly_energy_kwh = float(monthly_old.get("energy_kwh", 0)) + energy_increment_kwh
    monthly_co2_kg = float(monthly_old.get("co2_saved_kg", 0)) + co2_increment_kg

    monthly_ref.set({
        "month": month_key,
        "energy_kwh": round(monthly_energy_kwh, 6),
        "co2_saved_kg": round(monthly_co2_kg, 6),
        "co2_factor_kg_per_kwh": CO2_EMISSION_FACTOR,
        "last_increment_energy_kwh": round(energy_increment_kwh, 8),
        "last_increment_co2_kg": round(co2_increment_kg, 8),
        "last_updated": now
    })

    lifetime_ref = db.reference("HIRA/energy_stats/lifetime")
    lifetime_old = lifetime_ref.get() or {}

    lifetime_energy_kwh = float(lifetime_old.get("energy_kwh", 0)) + energy_increment_kwh
    lifetime_co2_kg = float(lifetime_old.get("co2_saved_kg", 0)) + co2_increment_kg

    lifetime_ref.set({
        "energy_kwh": round(lifetime_energy_kwh, 6),
        "co2_saved_kg": round(lifetime_co2_kg, 6),
        "co2_factor_kg_per_kwh": CO2_EMISSION_FACTOR,
        "last_increment_energy_kwh": round(energy_increment_kwh, 8),
        "last_increment_co2_kg": round(co2_increment_kg, 8),
        "last_updated": now
    })

    return {
        "today_key": today_key,
        "month_key": month_key,
        "energy_increment_kwh": energy_increment_kwh,
        "co2_increment_kg": co2_increment_kg,
        "daily_energy_kwh": daily_energy_kwh,
        "daily_co2_kg": daily_co2_kg,
        "monthly_energy_kwh": monthly_energy_kwh,
        "monthly_co2_kg": monthly_co2_kg,
        "lifetime_energy_kwh": lifetime_energy_kwh,
        "lifetime_co2_kg": lifetime_co2_kg
    }

# ==============================
# Main Loop
# ==============================
while True:
    try:
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
        response = requests.get(url, timeout=10)
        data = response.json()

        if "feeds" not in data or len(data["feeds"]) == 0:
            print("No ThingSpeak data found")
            time.sleep(LOOP_INTERVAL_SEC)
            continue

        df = pd.DataFrame(data["feeds"])

        for i in range(1, 9):
            df[f"field{i}"] = pd.to_numeric(df[f"field{i}"], errors="coerce")

        df = df.rename(columns={
            "field1": "voltage",
            "field2": "current",
            "field3": "power",
            "field4": "temperature",
            "field5": "humidity",
            "field6": "lux",
            "field7": "ghi",
            "field8": "relay_count"
        })

        df = df.dropna()

        if len(df) == 0:
            print("No valid data")
            time.sleep(LOOP_INTERVAL_SEC)
            continue

        latest = df.iloc[-1:]

        voltage = float(latest["voltage"].iloc[0])
        current = float(latest["current"].iloc[0])
        actual = float(latest["power"].iloc[0])
        temperature = float(latest["temperature"].iloc[0])
        humidity = float(latest["humidity"].iloc[0])
        lux = float(latest["lux"].iloc[0])
        ghi = float(latest["ghi"].iloc[0])
        relay_count = int(latest["relay_count"].iloc[0])

        now_dt = datetime.now(ist)
        now = now_dt.strftime("%d-%m-%Y %I:%M:%S %p")

        energy_stats = update_energy_stats(actual, now_dt, now)

        X = latest[[
            "voltage",
            "current",
            "temperature",
            "humidity",
            "lux",
            "ghi",
            "relay_count"
        ]]

        if actual < 4.0:
            mode = "LOW_POWER_ANN"
            X_scaled = low_scaler.transform(X)
            predicted = float(low_model.predict(X_scaled, verbose=0)[0][0])
        else:
            mode = "HIGH_POWER_ANN"
            X_scaled = high_scaler.transform(X)
            predicted = float(high_model.predict(X_scaled, verbose=0)[0][0])

        if predicted <= 0:
            efficiency = 0.0
            status = "Prediction Error"
        else:
            efficiency = (actual / predicted) * 100
            efficiency = float(np.clip(efficiency, 0, 150))
            status = get_status(efficiency, mode)

        fault_data = detect_fault(
            actual,
            predicted,
            efficiency,
            voltage,
            current,
            lux,
            ghi,
            relay_count
        )

        ai_data = {
            "actual_power": round(actual, 3),
            "predicted_power": round(predicted, 3),
            "efficiency": round(efficiency, 2),
            "status": status,
            "mode": mode,

            "alarm_level": fault_data["alarm_level"],
            "fault_type": fault_data["fault_type"],
            "fault_message": fault_data["fault_message"],
            "soiling_loss": round(fault_data["soiling_loss"], 2),
            "performance_ratio": round(fault_data["performance_ratio"], 3),
            "power_loss": round(fault_data["power_loss"], 3),

            "daily_energy_kwh": round(energy_stats["daily_energy_kwh"], 6),
            "daily_co2_saved_kg": round(energy_stats["daily_co2_kg"], 6),

            "monthly_energy_kwh": round(energy_stats["monthly_energy_kwh"], 6),
            "monthly_co2_saved_kg": round(energy_stats["monthly_co2_kg"], 6),

            "lifetime_energy_kwh": round(energy_stats["lifetime_energy_kwh"], 6),
            "lifetime_co2_saved_kg": round(energy_stats["lifetime_co2_kg"], 6),

            "energy_increment_kwh": round(energy_stats["energy_increment_kwh"], 8),
            "co2_increment_kg": round(energy_stats["co2_increment_kg"], 8),
            "co2_factor_kg_per_kwh": CO2_EMISSION_FACTOR,

            "timestamp": now
        }

        db.reference("HIRA/ai").set(ai_data)

        # ==============================
        # Store AI History
        # ==============================
        history_ref = db.reference("HIRA/ai_history")

        history_ref.push({
            "actual_power": round(actual, 3),
            "predicted_power": round(predicted, 3),
            "efficiency": round(efficiency, 2),
            "alarm_level": fault_data["alarm_level"],
            "fault_type": fault_data["fault_type"],

            "daily_energy_kwh": round(energy_stats["daily_energy_kwh"], 6),
            "daily_co2_saved_kg": round(energy_stats["daily_co2_kg"], 6),

            "timestamp": now
        })

        # ==============================
        # Keep Last 100 History Points
        # ==============================
        snapshot = history_ref.order_by_key().limit_to_last(120).get()

        if snapshot and len(snapshot) > 100:
            keys = list(snapshot.keys())
            keys.sort()

            for k in keys[:len(keys) - 100]:
                history_ref.child(k).delete()

        # ==============================
        # Daily Analytics
        # ==============================
        daily_analytics_ref = db.reference("HIRA/daily_analytics").child(energy_stats["today_key"])
        old = daily_analytics_ref.get()

        if old is None:
            count = 1
            avg_efficiency = efficiency
            avg_actual_power = actual
            avg_predicted_power = predicted
            max_actual_power = actual
            fault_count = 0
            alarm_count = 0
        else:
            count = int(old.get("count", 0)) + 1
            avg_efficiency = ((float(old.get("avg_efficiency", 0)) * (count - 1)) + efficiency) / count
            avg_actual_power = ((float(old.get("avg_actual_power", 0)) * (count - 1)) + actual) / count
            avg_predicted_power = ((float(old.get("avg_predicted_power", 0)) * (count - 1)) + predicted) / count
            max_actual_power = max(float(old.get("max_actual_power", 0)), actual)

            fault_count = int(old.get("fault_count", 0))
            alarm_count = int(old.get("alarm_count", 0))

        if fault_data["fault_type"] != "No Fault":
            fault_count += 1

        if fault_data["alarm_level"] in ["ALARM", "CRITICAL"]:
            alarm_count += 1

        daily_analytics_ref.set({
            "date": energy_stats["today_key"],
            "count": count,

            "avg_efficiency": round(avg_efficiency, 2),
            "avg_actual_power": round(avg_actual_power, 3),
            "avg_predicted_power": round(avg_predicted_power, 3),
            "max_actual_power": round(max_actual_power, 3),

            "daily_energy_kwh": round(energy_stats["daily_energy_kwh"], 6),
            "daily_co2_saved_kg": round(energy_stats["daily_co2_kg"], 6),

            "monthly_energy_kwh": round(energy_stats["monthly_energy_kwh"], 6),
            "monthly_co2_saved_kg": round(energy_stats["monthly_co2_kg"], 6),

            "lifetime_energy_kwh": round(energy_stats["lifetime_energy_kwh"], 6),
            "lifetime_co2_saved_kg": round(energy_stats["lifetime_co2_kg"], 6),

            "co2_factor_kg_per_kwh": CO2_EMISSION_FACTOR,

            "fault_count": fault_count,
            "alarm_count": alarm_count,

            "last_status": status,
            "last_alarm_level": fault_data["alarm_level"],
            "last_fault_type": fault_data["fault_type"],
            "last_updated": now
        })

        # ==============================
        # Premium Alert + FCM Notification
        # ==============================
        if fault_data["alarm_level"] in ["ALARM", "CRITICAL"]:

            alerts_ref = db.reference("HIRA/alerts")

            alerts_ref.push({
                "title": "🚨 HIRA Critical Energy Alert",
                "message": fault_data["fault_type"],
                "details": fault_data["fault_message"],
                "alarm_level": fault_data["alarm_level"],
                "time": now
            })

            # ==============================
            # Keep Last 25 Alert Points
            # ==============================
            alert_snapshot = alerts_ref.order_by_key().limit_to_last(30).get()

            if alert_snapshot and len(alert_snapshot) > 25:
                alert_keys = list(alert_snapshot.keys())
                alert_keys.sort()

                for k in alert_keys[:len(alert_keys) - 25]:
                    alerts_ref.child(k).delete()

            send_premium_notification(fault_data, now)

        print("Firebase Updated")
        print("Mode:", mode)
        print("Actual Power:", round(actual, 3), "W")
        print("Predicted Power:", round(predicted, 3), "W")
        print("Efficiency:", round(efficiency, 2), "%")
        print("Status:", status)
        print("Alarm Level:", fault_data["alarm_level"])
        print("Fault Type:", fault_data["fault_type"])
        print("Daily Energy:", round(energy_stats["daily_energy_kwh"], 6), "kWh")
        print("Daily CO2:", round(energy_stats["daily_co2_kg"], 6), "kg")
        print("Monthly Energy:", round(energy_stats["monthly_energy_kwh"], 6), "kWh")
        print("Monthly CO2:", round(energy_stats["monthly_co2_kg"], 6), "kg")
        print("Lifetime Energy:", round(energy_stats["lifetime_energy_kwh"], 6), "kWh")
        print("Lifetime CO2:", round(energy_stats["lifetime_co2_kg"], 6), "kg")
        print("Time:", now)
        print("------------------------")

        time.sleep(LOOP_INTERVAL_SEC)

    except Exception as e:
        print("Error:", e)
        time.sleep(10)

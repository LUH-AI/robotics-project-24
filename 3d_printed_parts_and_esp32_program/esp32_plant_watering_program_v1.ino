#include <Wire.h>
#include <VL53L0X.h>
#include <MPU6050.h>

// Sensor objects
VL53L0X distanceSensor;
MPU6050 mpu;

// Thresholds
#define DISTANCE_THRESHOLD 150   // 5 cm (VL53L0X gives mm)
#define ACCEL_THRESHOLD 0.33    // Minimal acceleration (g)
#define GYRO_THRESHOLD 30.0      // Minimal rotational acceleration (°/s)

// Output Pins
#define OUTPUT_LED1 12  // First output pin
#define OUTPUT_LED2 13  // First output pin
#define OUTPUT_vent 27  // Second output pin

// Global variables
int count = 0;   // Counter for how many times the conditions were met in a row
// Define variables to create some running average
float accelX = 0;  // ±2g scale
float accelY = 0;
float accelZ = 0;

float gyroX = 0;  // ±250°/s scale
float gyroY = 0;
float gyroZ = 0;

void setup() {
    Serial.begin(115200);
    Wire.begin();  // Default SDA=21, SCL=22 on ESP32

    // Initialize VL53L0X
    if (!distanceSensor.init()) {
        Serial.println("Failed to detect VL53L0X!");
        while (1);
    }
    distanceSensor.setTimeout(500);
    distanceSensor.startContinuous();

    // Initialize MPU6050
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while (1);
    }

    // Set output pins as OUTPUT
    pinMode(OUTPUT_LED1, OUTPUT);
    pinMode(OUTPUT_LED2, OUTPUT);
    pinMode(OUTPUT_vent, OUTPUT);
    
    // Ensure outputs are OFF initially
    digitalWrite(OUTPUT_LED1, LOW);
    digitalWrite(OUTPUT_LED2, LOW);
    digitalWrite(OUTPUT_vent, LOW);

    Serial.println("Sensors Initialized. Starting...");

    count = 0;
}

void loop() {
    // Read distance
    int distance = distanceSensor.readRangeContinuousMillimeters();
    
    // Read acceleration and gyroscope
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Convert raw values to real-world units
    accelX = accelX * 0.5 + 0.5 * ax / 16384.0;  // ±2g scale
    accelY = accelY * 0.5 + 0.5 * ay / 16384.0;
    accelZ = accelZ * 0.5 + 0.5 * az / 16384.0;
    
    gyroX = gyroX * 0.5 + 0.5 * gx / 131.0;  // ±250°/s scale
    gyroY = gyroY * 0.5 + 0.5 * gy / 131.0;
    gyroZ = gyroZ * 0.5 + 0.5 * gz / 131.0;

    // Calculate inclination (device horizontal when az ≈ 1g and ax, ay ≈ 0g)
    bool isHorizontal = (abs(accelX) < ACCEL_THRESHOLD) && (abs(accelY) < ACCEL_THRESHOLD) && (abs(accelZ - 1.0) < ACCEL_THRESHOLD);

    // Check for minimal rotation
    bool isStable = (abs(gyroX) < GYRO_THRESHOLD) && (abs(gyroY) < GYRO_THRESHOLD) && (abs(gyroZ) < GYRO_THRESHOLD);

    // Check conditions
    if (distance < DISTANCE_THRESHOLD && isHorizontal && isStable) {
        Serial.println("ACTION");
        // Activate the first status LED each time, all criterias are met
        digitalWrite(OUTPUT_LED1, HIGH);
        // Count the number of met conditiuons in a row
        count+=1;
        // If more than ten in a row, activate the plant watering mechansim/ action
        if count>=10{
          // Signal the action via the second LED and activate the vent
          digitalWrite(OUTPUT_LED2, HIGH);
          digitalWrite(OUTPUT_vent, HIGH);
          delay(4000); // Keep the vent opened for 4 seconds 
          // it takes about additional 1.5s for the watering process to stop
          count = -150;
          // Deactivate the vent and the second status LED
          digitalWrite(OUTPUT_LED2, LOW);
          digitalWrite(OUTPUT_vent, LOW);          
        }else{
          digitalWrite(OUTPUT_LED2, LOW);
          digitalWrite(OUTPUT_vent, LOW);
        }
    } else {
        if (count>0){
          count = 0;
        }else{
          count+=1;
        }
        digitalWrite(OUTPUT_LED1, LOW);
        digitalWrite(OUTPUT_LED2, LOW);
        digitalWrite(OUTPUT_vent, LOW);
    }

    // Debug output
    Serial.print("Dist: "); Serial.print(distance); Serial.print(" mm, ");
    Serial.print("Accel (X,Y,Z): "); Serial.print(accelX); Serial.print(", "); Serial.print(accelY); Serial.print(", "); Serial.print(accelZ); Serial.print(" g, ");
    Serial.print("Gyro (X,Y,Z): "); Serial.print(gyroX); Serial.print(", "); Serial.print(gyroY); Serial.print(", "); Serial.print(gyroZ); Serial.println(" °/s");
    delay(100);  // 100ms loop
}

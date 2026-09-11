import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FireDetection(Base):
    __tablename__ = "fire_detections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    source = Column(String(50), default="camera") # camera, video, satellite
    fire_confidence = Column(Float, default=0.0)
    smoke_confidence = Column(Float, default=0.0)
    severity = Column(String(20), default="LOW") # LOW, MEDIUM, HIGH, CRITICAL
    risk_score = Column(Float, default=0.0) # 0 to 100
    risk_level = Column(String(20), default="LOW") # LOW, MEDIUM, HIGH, CRITICAL
    model = Column(String(100), default="fusion")
    media_reference = Column(Text, nullable=True) # filename, url, or path

class SatelliteEvent(Base):
    __tablename__ = "satellite_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    confidence = Column(Float, default=0.0) # 0 to 100 or nominal/high
    FRP = Column(Float, default=0.0) # Fire Radiative Power in MW
    satellite = Column(String(50), default="VIIRS") # MODIS, VIIRS
    source = Column(String(50), default="NASA FIRMS")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    location = Column(String(200), default="Unspecified Location")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    severity = Column(String(20), default="LOW")
    risk = Column(String(20), default="LOW")
    message = Column(Text, nullable=False)
    status = Column(String(20), default="ACTIVE") # ACTIVE, RESOLVED, DISMISSED

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    status = Column(String(20), default="ONLINE")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Datasets & NASA Earthdata Integration

## Public NASA Sources
1. **NASA FIRMS (Fire Information for Resource Management System)**
   - Sensor Suite: MODIS (Terra/Aqua) & VIIRS (S-NPP/NOAA-20)
   - Parameters: Latitude, Longitude, Acquisition Date/Time, Confidence, FRP (Fire Radiative Power in MW).
   - Ingestion: REST CSV API endpoint.

2. **Local Fallback (`MODE=MOCK`)**
   - Offline development fallback providing realistic sample records clearly tagged with `"is_mock": true` and `"source": "MOCK LOCAL NASA DATA"`.

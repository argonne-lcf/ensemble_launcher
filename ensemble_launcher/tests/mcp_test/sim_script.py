def sim(Temperature: float, Pressure: float) -> str:
    """
    Runs a CFD simulation at a given temperature and pressure
    """
    import time
    time.sleep(1.0)
    return f"Ran simulation at Temperature:{Temperature} and Pressure: {Pressure}"
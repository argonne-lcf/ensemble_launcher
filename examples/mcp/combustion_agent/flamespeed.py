def compute_flame_speed(P: float, T: float, phi: float) -> float:
    """
    Compute the 1D freely-propagating flame speed for methane/air mixture.

    Parameters:
        P (float): Pressure in atm
        T (float): Unburned gas temperature in K
        phi (float): Equivalence ratio

    Returns:
        float: Flame speed in m/s
    """
    import cantera as ct

    # Methane/air stoichiometry: CH4 + 2 O2 + 7.52 N2
    # For equivalence ratio phi: CH4:phi, O2:2, N2:7.52
    reactants = f'CH4:{phi}, O2:2, N2:7.52'
    width = 0.03  # m
    loglevel = 1

    gas = ct.Solution('gri30.yaml')
    gas.TPX = T, P*ct.one_atm, reactants

    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    f.transport_model = 'mixture-averaged'
    f.solve(loglevel=loglevel)

    return f.velocity[0]


if __name__ == "__main__":
    print(compute_flame_speed(1.0, 300.0, 0.5))
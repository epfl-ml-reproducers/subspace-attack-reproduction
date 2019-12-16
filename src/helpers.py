def boolean_string(s: str) -> bool:
    """
    Checks whether a boolean command line argument is `True` or `False`.

    Parameters
    ----------
    s: str
        The command line argument to be checked.

    Returns
    -------
    bool_argument: bool
        Whether the argument is `True` or `False`

    Raises
    ------
    ValueError
        If the input is not 'True' nor 'False'
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    
    return s == 'True'
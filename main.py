"""
Backward-compatible launcher.  The real CLI lives in allianceai.cli;
after `pip install -e .` you can also just run `allianceai AAPL`.
"""

from allianceai.cli import main

if __name__ == "__main__":
    main()

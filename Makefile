.PHONY: setup run test lint

setup:
\tpip install -r requirements.txt
\tpip install pytest pre-commit
\tpre-commit install

run:
\tstreamlit run streamlit_app.py

test:
\tpytest -q

lint:
\tpre-commit run --all-files

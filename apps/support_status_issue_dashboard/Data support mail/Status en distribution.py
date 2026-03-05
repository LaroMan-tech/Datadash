from __future__ import annotations

from pathlib import Path
import runpy
import traceback
import sys

import streamlit as st


APP_TITLE = "Groot Dashboard - Service + Support Mails"


def _apply_python314_typed_dict_compat() -> None:
    """Work around Altair TypedDict incompatibility on Python 3.14."""
    if sys.version_info < (3, 14):
        return
    try:
        import typing
        from typing_extensions import TypedDict as _TypedDictExt
    except Exception:
        return
    typing.TypedDict = _TypedDictExt  # type: ignore[assignment]


def run_external_streamlit_main(script_path: Path) -> None:
    if not script_path.exists():
        st.error(f"Bestand niet gevonden: {script_path}")
        return

    try:
        _apply_python314_typed_dict_compat()
        module_globals = runpy.run_path(str(script_path))
        main_fn = module_globals.get("main")
        if main_fn is None:
            st.error(f"Geen main() gevonden in: {script_path.name}")
            return

        # Parent app heeft set_page_config al gedaan; voorkom dubbele call.
        original_set_page_config = st.set_page_config
        st.set_page_config = lambda *args, **kwargs: None
        try:
            main_fn()
        finally:
            st.set_page_config = original_set_page_config
    except Exception:
        st.error(f"Fout bij laden van {script_path.name}")
        st.code(traceback.format_exc())


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Combinatie van service rapport data en support mail issue distribution.")

    this_file = Path(__file__).resolve()
    data_support_mail_dir = this_file.parent
    support_processen_dir = data_support_mail_dir.parent
    service_data_dir = support_processen_dir / "service rapport data"

    issue_script = data_support_mail_dir / "App data dash issue distribution.py"
    service_script = service_data_dir / "Startanalyse 2 backend 0.1 test met engage.py"

    tab_service, tab_issue = st.tabs(
        ["Service rapport dashboard", "Support mail issue distribution"]
    )

    with tab_service:
        run_external_streamlit_main(service_script)

    with tab_issue:
        run_external_streamlit_main(issue_script)


if __name__ == "__main__":
    main()

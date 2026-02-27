def setup_plutoserver():
    return {
        "command": ["sh", "binder/start_server.sh"],
        "environment": {
            "JSP_PORT": "{port}",
        },
        "timeout": 60,
        "launcher_entry": {
            "title": "Pluto.jl",
        },
    }

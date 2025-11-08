from flask import Flask
from flask_cors import CORS
try:
    from api.router import bp
except Exception:
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from api.router import bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(bp)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
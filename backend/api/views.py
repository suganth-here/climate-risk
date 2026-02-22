import json

import pandas as pd
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .logic import (
    DEFAULT_PROJECTION_START_YEAR,
    analyze_portfolio,
    evaluate_single_application,
    metadata_payload,
)


def _bad_request(message: str) -> JsonResponse:
    return JsonResponse({"error": message}, status=400)


def _method_not_allowed() -> JsonResponse:
    return JsonResponse({"error": "Method not allowed."}, status=405)


def metadata_view(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return _method_not_allowed()
    return JsonResponse(metadata_payload())


@csrf_exempt
def predict_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _method_not_allowed()
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return _bad_request("Invalid JSON payload.")

    required = ["latitude", "longitude", "tenure_years", "loan_amount"]
    missing = [k for k in required if k not in payload]
    if missing:
        return _bad_request(f"Missing required fields: {missing}")

    try:
        result = evaluate_single_application(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            tenure_years=int(payload["tenure_years"]),
            loan_amount=float(payload["loan_amount"]),
            property_amount=float(payload.get("property_amount", 0.0)),
            projection_start_year=int(payload.get("projection_start_year", DEFAULT_PROJECTION_START_YEAR)),
            property_id=str(payload.get("property_id", "98122")),
        )
    except Exception as exc:
        return _bad_request(f"Could not run prediction: {exc}")

    return JsonResponse(result)


@csrf_exempt
def portfolio_analyze_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _method_not_allowed()

    projection_start_year = DEFAULT_PROJECTION_START_YEAR
    portfolio_df = None

    if "file" in request.FILES:
        try:
            uploaded = request.FILES["file"]
            portfolio_df = pd.read_csv(uploaded)
            if "projection_start_year" in request.POST:
                projection_start_year = int(request.POST["projection_start_year"])
        except Exception as exc:
            return _bad_request(f"Could not read uploaded CSV: {exc}")
    else:
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return _bad_request("Invalid JSON payload.")
        rows = payload.get("rows")
        if rows is None:
            return _bad_request("Provide multipart 'file' or JSON 'rows'.")
        try:
            portfolio_df = pd.DataFrame(rows)
            projection_start_year = int(payload.get("projection_start_year", DEFAULT_PROJECTION_START_YEAR))
        except Exception as exc:
            return _bad_request(f"Could not parse portfolio rows: {exc}")

    if portfolio_df is None:
        return _bad_request("No portfolio data found.")

    try:
        result = analyze_portfolio(portfolio_df, projection_start_year=projection_start_year)
    except Exception as exc:
        return _bad_request(f"Could not process uploaded CSV: {exc}")

    return JsonResponse(result)

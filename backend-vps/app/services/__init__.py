"""Service layer — business logic. Talks to repositories (DB) and the model-server
(HTTP via ``model_client``); never touches the ORM or ``requests`` directly from
the API layer.
"""

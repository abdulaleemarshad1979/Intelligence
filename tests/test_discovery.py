"""Tests for ONVIF WS-Discovery, camera auto-discovery, and database ingestion."""

import pytest
from starlette.testclient import TestClient
from app.main import app
from app.discovery.onvif_scanner import ONVIFDiscoveryScanner, DiscoveredCamera
from app.database.repository import Repository
from app.database.models import CameraEntity


@pytest.fixture
def client():
    return TestClient(app)


def test_probe_match_xml_parsing():
    scanner = ONVIFDiscoveryScanner()

    sample_soap_xml = """<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
                   xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
                   xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
                   xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
      <soap:Header>
        <wsa:RelatesTo>uuid:12345678-abcd</wsa:RelatesTo>
      </soap:Header>
      <soap:Body>
        <d:ProbeMatches>
          <d:ProbeMatch>
            <wsa:EndpointReference>
              <wsa:Address>urn:uuid:7c037890-8888-11ec-b909-00123456789a</wsa:Address>
            </wsa:EndpointReference>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
            <d:Scopes>
              onvif://www.onvif.org/type/NetworkVideoTransmitter
              onvif://www.onvif.org/name/Hikvision
              onvif://www.onvif.org/hardware/DS-2CD2043G2-I
              onvif://www.onvif.org/mac/00:1A:2B:3C:4D:5E
            </d:Scopes>
            <d:XAddrs>http://192.168.1.188:80/onvif/device_service</d:XAddrs>
          </d:ProbeMatch>
        </d:ProbeMatches>
      </soap:Body>
    </soap:Envelope>"""

    cam = scanner._parse_ws_probe_match(sample_soap_xml, "192.168.1.188")
    assert cam is not None
    assert cam.ip_address == "192.168.1.188"
    assert cam.port == 80
    assert cam.manufacturer == "Hikvision"
    assert cam.model_name == "DS-2CD2043G2-I"
    assert cam.mac_address == "00:1A:2B:3C:4D:5E"
    assert "Streaming/Channels/101" in cam.rtsp_url


def test_scanner_discovers_fleet():
    scanner = ONVIFDiscoveryScanner()
    cams = scanner.scan_network(timeout=0.1, include_simulated_if_empty=True)
    assert len(cams) >= 2
    ips = [c.ip_address for c in cams]
    assert "192.168.1.120" in ips


def test_repository_discovery_and_approval():
    repo = Repository()
    cam_id = "CAM-TEST-DISCOVER-99"

    test_cam = CameraEntity(
        camera_id=cam_id,
        name="Auto Discovered Dahua Cam",
        latitude=16.90,
        longitude=82.20,
        zone="Unassigned",
        ip_address="192.168.1.99",
        rtsp_url="rtsp://192.168.1.99:554/live",
        manufacturer="Dahua",
        model_name="IPC-HDW2431T",
        discovery_status="DISCOVERED",
        is_active=False
    )
    assert repo.save_camera(test_cam) is True

    retrieved = repo.get_camera_by_id(cam_id)
    assert retrieved is not None
    assert retrieved.discovery_status == "DISCOVERED"
    assert retrieved.is_active is False

    # Officer approves the camera with friendly name and coordinates
    ok = repo.approve_discovered_camera(
        camera_id=cam_id,
        name="South Gate Ambulance Bay",
        latitude=16.9895,
        longitude=82.2480,
        zone="East Zone"
    )
    assert ok is True

    approved = repo.get_camera_by_id(cam_id)
    assert approved.name == "South Gate Ambulance Bay"
    assert approved.discovery_status == "APPROVED"
    assert approved.is_active is True
    assert approved.latitude == 16.9895


def test_api_discovery_endpoints(client):
    # 1. Trigger scan
    resp = client.post("/api/discovery/scan", json={"timeout_seconds": 0.1, "include_simulated_if_empty": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["found_count"] >= 2

    # 2. List discovered cameras
    resp_list = client.get("/api/discovery/cameras")
    assert resp_list.status_code == 200
    cams = resp_list.json()["cameras"]
    assert len(cams) >= 1

    # 3. Approve first discovered camera
    first_cam = cams[0]
    approve_resp = client.post("/api/discovery/approve", json={
        "camera_id": first_cam["camera_id"],
        "name": "North Gate Entrance Approved",
        "latitude": 16.9912,
        "longitude": 82.2490,
        "zone": "North Zone"
    })
    assert approve_resp.status_code == 200
    app_data = approve_resp.json()
    assert app_data["status"] == "SUCCESS"
    assert app_data["camera"]["status"] == "APPROVED"

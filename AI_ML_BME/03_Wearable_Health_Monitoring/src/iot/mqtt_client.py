"""MQTT client for IoT communication."""

import json
import time
import logging
from typing import Optional, Callable, Dict, Any, List
import paho.mqtt.client as mqtt
from src.config.settings import MQTT_CONFIG, DEVICE_CONFIG

logger = logging.getLogger(__name__)


class MQTTClient:
    """MQTT client for publishing and subscribing to health data."""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize MQTT client.
        
        Args:
            client_id: Unique client ID. If None, uses device_id from config.
            broker: MQTT broker address
            port: MQTT broker port
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.client_id = client_id or DEVICE_CONFIG["device_id"]
        self.broker = broker or MQTT_CONFIG["broker"]
        self.port = port or MQTT_CONFIG["port"]
        self.username = username or MQTT_CONFIG["username"]
        self.password = password or MQTT_CONFIG["password"]
        self.topic_prefix = MQTT_CONFIG["topic_prefix"]
        self.qos = MQTT_CONFIG["qos"]
        self.keepalive = MQTT_CONFIG["keepalive"]
        
        self.client = None
        self.is_connected = False
        self.message_callbacks: Dict[str, List[Callable]] = {}
        
        self._setup_client()
    
    def _setup_client(self):
        """Set up MQTT client with callbacks."""
        self.client = mqtt.Client(client_id=self.client_id)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        
        # Set credentials if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        
        # Enable logging
        self.client.enable_logger(logger)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker."""
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
            # Resubscribe to topics
            for topic in self.message_callbacks.keys():
                self.client.subscribe(topic, qos=self.qos)
        else:
            self.is_connected = False
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker."""
        self.is_connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message is received."""
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
            logger.debug(f"Received message on topic {topic}: {payload}")
            
            # Call registered callbacks for this topic
            if topic in self.message_callbacks:
                for callback in self.message_callbacks[topic]:
                    try:
                        callback(topic, payload)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message on topic {topic}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback when message is published."""
        logger.debug(f"Message published with mid: {mid}")
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            self.client.connect(self.broker, self.port, self.keepalive)
            self.client.loop_start()
            # Wait a bit for connection to establish
            time.sleep(0.5)
            return self.is_connected
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from MQTT broker")
    
    def publish(self, topic_suffix: str, payload: Dict[str, Any], retain: bool = False) -> bool:
        """
        Publish message to MQTT topic.
        
        Args:
            topic_suffix: Topic suffix (will be prefixed with topic_prefix/device_id)
            payload: Message payload (will be JSON encoded)
            retain: Whether to retain the message
        
        Returns:
            True if published successfully, False otherwise
        """
        if not self.is_connected:
            logger.warning("Not connected to MQTT broker. Attempting to connect...")
            if not self.connect():
                return False
        
        topic = f"{self.topic_prefix}/{self.client_id}/{topic_suffix}"
        
        try:
            message = json.dumps(payload)
            result = self.client.publish(topic, message, qos=self.qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published to topic {topic}")
                return True
            else:
                logger.error(f"Failed to publish to topic {topic}. Return code: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    def subscribe(self, topic_suffix: str, callback: Callable) -> bool:
        """
        Subscribe to MQTT topic.
        
        Args:
            topic_suffix: Topic suffix (will be prefixed with topic_prefix)
            callback: Callback function to call when message is received
                    Function signature: callback(topic: str, payload: dict)
        
        Returns:
            True if subscribed successfully, False otherwise
        """
        if not self.is_connected:
            logger.warning("Not connected to MQTT broker. Attempting to connect...")
            if not self.connect():
                return False
        
        topic = f"{self.topic_prefix}/{topic_suffix}"
        
        try:
            result = self.client.subscribe(topic, qos=self.qos)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                # Register callback
                if topic not in self.message_callbacks:
                    self.message_callbacks[topic] = []
                self.message_callbacks[topic].append(callback)
                logger.info(f"Subscribed to topic {topic}")
                return True
            else:
                logger.error(f"Failed to subscribe to topic {topic}. Return code: {result[0]}")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to topic: {e}")
            return False
    
    def unsubscribe(self, topic_suffix: str, callback: Optional[Callable] = None):
        """
        Unsubscribe from MQTT topic.
        
        Args:
            topic_suffix: Topic suffix
            callback: Specific callback to remove. If None, removes all callbacks for topic.
        """
        topic = f"{self.topic_prefix}/{topic_suffix}"
        
        try:
            if callback is None:
                # Remove all callbacks and unsubscribe
                if topic in self.message_callbacks:
                    del self.message_callbacks[topic]
                self.client.unsubscribe(topic)
            else:
                # Remove specific callback
                if topic in self.message_callbacks:
                    if callback in self.message_callbacks[topic]:
                        self.message_callbacks[topic].remove(callback)
                    if not self.message_callbacks[topic]:
                        del self.message_callbacks[topic]
                        self.client.unsubscribe(topic)
            logger.info(f"Unsubscribed from topic {topic}")
        except Exception as e:
            logger.error(f"Error unsubscribing from topic: {e}")


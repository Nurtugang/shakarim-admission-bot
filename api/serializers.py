from rest_framework import serializers

class KnowledgeBaseSerializer(serializers.Serializer):
    category = serializers.CharField(max_length=255)
    text = serializers.CharField()

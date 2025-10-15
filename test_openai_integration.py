#!/usr/bin/env python3
"""
Simple test script to verify OpenAI integration works.
"""

import os
from dotenv import load_dotenv
from synthetic_questions import SyntheticQuestionGenerator

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test basic OpenAI integration."""
    print("Testing OpenAI integration...")
    
    try:
        # Initialize generator
        generator = SyntheticQuestionGenerator()
        print("✅ Generator initialized successfully")
        
        # Test a simple LLM call
        test_prompt = "Generate 3 simple test questions about Python programming."
        
        print("🤖 Testing LLM call...")
        response = generator.llm.invoke(test_prompt)
        response_text = response.content
        
        print(f"✅ LLM response received ({len(response_text)} characters)")
        print("\nSample response:")
        print("-" * 50)
        print(response_text)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_integration()
    if success:
        print("\n🎉 OpenAI integration test PASSED!")
    else:
        print("\n💥 OpenAI integration test FAILED!")
    exit(0 if success else 1)
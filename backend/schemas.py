from typing import Optional, Dict, Any
from pydantic import BaseModel
 
class DocumentInfo(BaseModel):
    document_type: Optional[str] = None
    card_number: Optional[str] = None
    issuing_authority: Optional[str] = None
    municipality: Optional[str] = None
 
 
class PersonalInfo(BaseModel):
    name: Optional[str] = None
    father_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    hair_color: Optional[str] = None
    eye_color: Optional[str] = None
    clothing: Optional[str] = None
 
 
class AddressInfo(BaseModel):
    house_flat_no: Optional[str] = None
    street_area: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    country: Optional[str] = None
 
 
class ExtractionResponse(BaseModel):
    document_info: DocumentInfo = DocumentInfo()
    personal_info: PersonalInfo = PersonalInfo()
    address_info: AddressInfo = AddressInfo()
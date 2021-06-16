from django import forms
              

class select_check(forms.Form):
    country= forms.CharField(label='country', widget=forms.Select())
    state= forms.CharField(label='state', widget=forms.Select())
    city= forms.CharField(label='city', widget=forms.Select())
 
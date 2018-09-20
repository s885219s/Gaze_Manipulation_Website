from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class uploadPhotoForm(FlaskForm):
    photo = FileField('Updata Photo', validators=[FileRequired(),FileAllowed(['jpg','png','jpeg'], 'jpg and png only!')])
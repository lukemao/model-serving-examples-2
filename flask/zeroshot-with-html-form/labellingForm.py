from flask_wtf import FlaskForm
from wtforms import TextAreaField , StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, Length

class LabellingForm(FlaskForm):
    inputText           = TextAreaField('Text to be labelled', default='I am not unhappy about the service',
                                        validators=[DataRequired(), Length(min=2)])
    inputLabels         = StringField('Labels(in csv, e.g. happy,unhappy,neutral)', default='happy,unhappy,neutral', 
                                      validators=[DataRequired(), Length(min=2)])
    model               = SelectField('Model', choices=[('sb', 'deepset/sentence_bert'),
                                                        ('st', 'SentenceTransformer/bert-base-nli-mean-tokens')])
    runLabelling        = SubmitField('Submit', render_kw={"onclick": "loading()"})
    results             = TextAreaField('Results')
    
    
        

